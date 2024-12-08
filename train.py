import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from datetime import datetime
from pathlib import Path

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # Robust image conversion
        try:
            # Ensure image is a numpy array with correct dtype
            if not isinstance(image, np.ndarray):
                image = np.array(image, dtype=np.float32)
            
            # Ensure 3-channel color image
            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)
            
            # Transpose and normalize
            image = image.transpose((2, 0, 1))  # Convert to (C, H, W)
            image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]
            image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        except Exception as e:
            print(f"Error converting image to tensor: {e}")
            raise

        # Robust tensor conversion for boxes and labels
        if not isinstance(target['boxes'], torch.Tensor):
            target['boxes'] = torch.as_tensor(np.array(target['boxes'], dtype=np.float32))

        if not isinstance(target['labels'], torch.Tensor):
            target['labels'] = torch.as_tensor(np.array(target['labels'], dtype=np.int64))

        return image, target

def get_transform():
    return Compose([ToTensor()])

class RCNNDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transforms=None):       
        self.annotations = []
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Robust directory and file validation
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")
            
        if not os.path.exists(annotation_file):
            raise ValueError(f"Annotation file does not exist: {annotation_file}")

        # Annotation parsing with more robust error handling
        try:
            self._parse_annotations(annotation_file)
        except Exception as e:
            print(f"Error parsing annotations: {e}")
            raise

        # Define label mapping with more flexibility
        self.label_to_idx = {
            "Car": 1, "Pedestrian": 2, "Van": 3, "Cyclist": 4, 
            "Truck": 5, "Misc": 6, "Tram": 7, "Person_Sitting": 8, 
            "DontCare": 9, "Unknown": 10
        }

    def _parse_annotations(self, annotation_file):
        current_image = None
        current_boxes = []
        current_labels = []
        
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Processing annotations"):
            try:
                parts = line.strip().split(',')
                img_path = parts[0]
                
                # Normalize image path
                if img_path.startswith('RCNN_Dataset/'):
                    img_path = img_path[len('RCNN_Dataset/'):]
                
                full_img_path = os.path.join(self.root_dir, 'RCNN_Dataset', img_path)
                
                # Skip if image does not exist
                if not os.path.exists(full_img_path):
                    print(f"Warning: Image not found: {full_img_path}")
                    continue
                
                # Image change detection
                if current_image != img_path and current_image is not None:
                    if current_boxes:
                        self.annotations.append({
                            'image': current_image,
                            'boxes': current_boxes,
                            'labels': current_labels
                        })
                    current_boxes = []
                    current_labels = []
                
                current_image = img_path
                
                # Robust bbox parsing
                bbox = [float(x) for x in parts[1:5]]
                x1, y1, x2, y2 = bbox
                
                # Validate box coordinates
                if x1 >= x2 or y1 >= y2:
                    print(f"Warning: Invalid box coordinates in {img_path}: {bbox}")
                    continue
                
                current_boxes.append(bbox)
                current_labels.append(parts[5])
            
            except (ValueError, IndexError) as e:
                print(f"Warning: Invalid annotation format in line: {line.strip()}")
                continue
        
        # Add last image
        if current_image is not None and current_boxes:
            self.annotations.append({
                'image': current_image,
                'boxes': current_boxes,
                'labels': current_labels
            })
        
        print(f"Successfully loaded {len(self.annotations)} valid images")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, 'RCNN_Dataset', ann['image'])
        
        # Robust image loading
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert boxes and labels to tensors with error handling
        try:
            boxes = torch.as_tensor(np.array(ann['boxes'], dtype=np.float32))
            labels = torch.as_tensor(
                [self.label_to_idx.get(label, 10) for label in ann['labels']], 
                dtype=torch.int64
            )
        except Exception as e:
            print(f"Error converting annotations: {e}")
            raise

        # Additional metadata
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

class MetricTracker:
    def __init__(self, save_dir='training_metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'epoch': [], 'avg_loss': [], 'learning_rate': [],
            'loss_classifier': [], 'loss_box_reg': [],
            'loss_objectness': [], 'loss_rpn_box_reg': []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def update(self, epoch, avg_loss, lr, loss_dict):
        self.metrics['epoch'].append(epoch)
        self.metrics['avg_loss'].append(avg_loss)
        self.metrics['learning_rate'].append(lr)

        for loss_name in ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']:
            self.metrics[loss_name].append(loss_dict[loss_name].item())

    def save(self):
        df = pd.DataFrame(self.metrics)
        csv_path = self.save_dir / f'training_metrics_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved to {csv_path}")
        return df
    
def train_one_epoch(model, optimizer, data_loader, device, epoch, metric_tracker, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f'Epoch {epoch+1}', leave=True)
    
    for batch_idx, (images, targets) in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        
        # Scale loss and perform backward pass
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        running_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({
            'loss': f'{running_loss:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        if batch_idx % 10 == 0:
            metric_tracker.update(
                epoch=epoch,
                avg_loss=running_loss,
                lr=optimizer.param_groups[0]['lr'],
                loss_dict=loss_dict
            )
       
    return running_loss

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
            # Calculate accuracy
            for i, target in enumerate(targets):
                pred_boxes = model.roi_heads.box_predictor(model.roi_heads.box_encoder(images[i], target))[0]
                pred_labels = pred_boxes.get_field("labels")
                gt_labels = target["labels"]
                total_correct += (pred_labels == gt_labels).sum().item()
                total_samples += len(gt_labels)
                
    val_loss = total_loss / len(data_loader)
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy

def train_model(train_dataset, val_dataset, num_classes, num_epochs=50):
    print("✓ Initializing training...")

    # Enable TF32 for better performance on T4
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using {device.type.upper()}")

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    metric_tracker = MetricTracker()

    # Optimize data loading
    batch_size = 32  # Increased batch size for T4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),  # Reduced for Colab
        pin_memory=True,
        num_workers=1, # Faster data transfer to GPU
        prefetch_factor=2,
        persistent_workers=True
    )

    # Model setup with optimization flags
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.to(device)

    # Optimize memory usage
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad_(False)

    # Optimizer setup with more aggressive learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)

    # Cosine annealing scheduler for better convergence
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_loss = float('inf')
    print("\n" + "="*50 + "\nTraining Progress:\n" + "="*50)

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            metric_tracker, scaler
        )
        lr_scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'best_model.pth')

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

    metric_tracker.save()
    torch.save(model.state_dict(), 'final_rcnn_model.pth')
    print("\n✓ Training complete!")

    return model

def main():

    torch.cuda.empty_cache()

    root_dir = os.path.abspath("/home/usd.local/ujjwal.bhatta/ComputerVision-RCNN")
    train_annotation_file = "/home/usd.local/ujjwal.bhatta/ComputerVision-RCNN/train_annotation.txt"
    val_annotation_file = "/home/usd.local/ujjwal.bhatta/ComputerVision-RCNN/test_annotation.txt"
    
    print("Initializing datasets...")
    train_dataset = RCNNDataset(
        train_annotation_file,
        root_dir,
        transforms=get_transform()
    )
    
    val_dataset = RCNNDataset(
        val_annotation_file,
        root_dir,
        transforms=get_transform()
    )
    
    print(f"✓ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
    print(f"✓ Number of classes: {len(train_dataset.label_to_idx)}")
    
    model = train_model(train_dataset, val_dataset,len(train_dataset.label_to_idx), num_epochs=50)
    print("Training complete")

if __name__ == "__main__":
    main()

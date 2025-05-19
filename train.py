import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders, CrossValidationMode, TemporalWindowDataset
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN
import cv2
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in segmentation tasks.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the target class.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class. If None, uses inverse frequency.
            gamma: Focusing parameter. Higher values down-weight easy examples more.
            reduction: 'mean' or 'sum' for the loss reduction method.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7  # For numerical stability
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape [B, C, H, W]
            targets: Ground truth labels of shape [B, H, W]
        """
        # Convert inputs to probabilities using softmax
        probs = torch.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Calculate focal loss
        focal_weight = (1 - probs).pow(self.gamma)
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Reshape alpha to match the spatial dimensions
            alpha = self.alpha.view(1, -1, 1, 1)  # [1, C, 1, 1]
            focal_weight = alpha * focal_weight
        
        # Calculate the final loss
        loss = -focal_weight * targets_one_hot * torch.log(probs + self.eps)
        
        # Sum over classes and average over spatial dimensions
        loss = loss.sum(dim=1)  # [B, H, W]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    """
    Combined loss function using Focal Loss and Soft IoU Loss.
    """
    def __init__(self, alpha=None, gamma=2.0, iou_weight=0.3):
        """
        Args:
            alpha: Weighting factor for each class in Focal Loss
            gamma: Focusing parameter for Focal Loss
            iou_weight: Weight for IoU loss component (0-1)
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.iou_weight = iou_weight
        
    def soft_iou_loss(self, preds, targets, eps=1e-7):
        """
        Calculate soft IoU loss for the positive class.
        
        Args:
            preds: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            eps: Small value to avoid division by zero
            
        Returns:
            Soft IoU loss for the positive class
        """
        # Get probability for positive class
        preds = torch.softmax(preds, dim=1)
        pos_preds = preds[:, 1]  # (B, H, W)
        
        # Create one-hot encoding of targets
        pos_targets = (targets == 1).float()  # (B, H, W)
        
        # Calculate intersection and union
        intersection = (pos_preds * pos_targets).sum(dim=(1, 2))
        union = (pos_preds + pos_targets - pos_preds * pos_targets).sum(dim=(1, 2))
        
        # Calculate IoU
        iou = (intersection + eps) / (union + eps)
        
        # Return negative IoU (since we want to maximize IoU)
        return 1 - iou.mean()
    
    def forward(self, preds, targets):
        """
        Calculate combined loss.
        
        Args:
            preds: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Combined loss value
        """
        # Calculate focal loss
        focal = self.focal_loss(preds, targets)
        
        # Calculate IoU loss
        iou = self.soft_iou_loss(preds, targets)
        
        # Combine losses
        loss = (1 - self.iou_weight) * focal + self.iou_weight * iou
        
        return loss

# Create a filtered dataset class that only loads frames 0-49
class FilteredTemporalWindowDataset(TemporalWindowDataset):
    """
    Dataset that loads only the first 50 frames (0-49) from the original dataset.
    """
    def _load_data(self):
        """
        Override the _load_data method to only load the first 50 frames.
        """
        # This assumes data is organized with frames in frames/ and labels in labels/
        frames_dir = os.path.join(self.data_dir, "frames")
        labels_dir = os.path.join(self.data_dir, "labels")
        
        # Check if directories exist
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
            
        # List and sort files to ensure correct order
        supported_extensions = ['.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
        
        frame_files = []
        for ext in supported_extensions:
            frame_files.extend(sorted([f for f in os.listdir(frames_dir) if f.endswith(ext)]))
        
        label_files = []
        for ext in supported_extensions:
            label_files.extend(sorted([f for f in os.listdir(labels_dir) if f.endswith(ext)]))
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_dir}. "
                            f"Supported formats: {', '.join(supported_extensions)}")
        if not label_files:
            raise ValueError(f"No label files found in {labels_dir}. "
                            f"Supported formats: {', '.join(supported_extensions)}")
            
        # Limit to only frames 0-49
        frame_files = frame_files[:50]
        label_files = label_files[:50]
        
        print(f"Found {len(frame_files)} frame files and {len(label_files)} label files")
        print(f"Using only frames 0-49 for training/validation")
        
        frames = []
        labels = []
        
        for frame_file, label_file in zip(frame_files, label_files):
            frame_path = os.path.join(frames_dir, frame_file)
            label_path = os.path.join(labels_dir, label_file)
            
            # Load file based on extension
            if frame_file.endswith('.npy'):
                frame = np.load(frame_path)
            else:
                # Use OpenCV for image files
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise ValueError(f"Failed to load frame: {frame_path}")
            
            if label_file.endswith('.npy'):
                label = np.load(label_path)
            else:
                # Use OpenCV for image files
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label is None:
                    raise ValueError(f"Failed to load label: {label_path}")
            
            frames.append(frame)
            labels.append(label)
        
        print(f"Loaded {len(frames)} frames and {len(labels)} labels")
            
        return frames, labels

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths.
    
    Args:
        batch: List of (data, target) tuples from the dataset
        
    Returns:
        Batched data and targets with proper padding
    """
    # Separate data and targets
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Get the maximum sequence length in this batch
    max_seq_len = max([x.size(0) for x in data])
    
    # Get the dimensions of data
    batch_size = len(data)
    height = data[0].size(1)
    width = data[0].size(2)
    
    # Create a tensor to hold the padded data
    padded_data = torch.zeros(batch_size, max_seq_len, height, width)
    
    # Fill in the data with padding at the front
    for i, x in enumerate(data):
        seq_len = x.size(0)
        padded_data[i, -seq_len:] = x
    
    # Stack targets and convert from [0, 255] to [0, 1] for binary segmentation
    processed_targets = []
    for target in targets:
        # Convert any non-zero value to 1 for binary segmentation
        binary_target = (target > 0).long()
        processed_targets.append(binary_target)
    
    targets = torch.stack(processed_targets)
    
    return padded_data, targets

def parse_args():
    parser = argparse.ArgumentParser(description='Train defect detection models')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to output directory')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['cnn3d', 'vit', 'mamba', 'all'], 
                        help='Model type to train')
    
    # Dataset parameters
    parser.add_argument('--window_size', type=int, default=128, help='Window size for patches')
    parser.add_argument('--window_step', type=int, default=16, help='Step size for sliding window')
    parser.add_argument('--max_seq_length', type=int, default=5, help='Maximum sequence length')
    parser.add_argument('--cross_val', type=str, default='top', 
                        choices=['top', 'left', 'right', 'bottom', 'none'],
                        help='Cross-validation mode')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda or cpu)')
    
    return parser.parse_args()

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total_pixels = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * data.size(0)
        
        # Calculate accuracy for segmentation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total_pixels += target.numel()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100.0 * correct / total_pixels if total_pixels > 0 else 0.0
        })
    
    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100.0 * correct / total_pixels if total_pixels > 0 else 0.0
    
    return epoch_loss, epoch_acc

def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Validation')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Update statistics
            running_loss += loss.item() * data.size(0)
            
            # Calculate accuracy for segmentation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total_pixels += target.numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total_pixels if total_pixels > 0 else 0.0
            })
    
    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100.0 * correct / total_pixels if total_pixels > 0 else 0.0
    
    return epoch_loss, epoch_acc

def calculate_class_weights(data_loader):
    """
    Calculate class weights based on the distribution of classes in the dataset.
    Uses inverse frequency with a square root scaling to prevent extreme weights.
    
    Args:
        data_loader: DataLoader containing the dataset
        
    Returns:
        torch.Tensor: Class weights normalized to sum to 1
    """
    # Initialize counters for each class
    class_counts = torch.zeros(2)  # Binary segmentation
    
    # Count occurrences of each class
    for _, targets in data_loader:
        # Count occurrences of each class
        unique, counts = torch.unique(targets, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[u] += c
    
    # Calculate class weights using inverse frequency with square root scaling
    total_samples = class_counts.sum()
    class_weights = torch.sqrt(total_samples / class_counts)  # Square root scaling
    
    # Normalize weights to sum to 1
    class_weights = class_weights / class_weights.sum()
    
    print(f"Class distribution: {class_counts.tolist()}")
    print(f"Raw class weights: {(total_samples / class_counts).tolist()}")
    print(f"Square root scaled weights: {class_weights.tolist()}")
    
    return class_weights

def train_model(model_name, model, train_loader, val_loader, args):
    print(f"\n=== Training {model_name} ===")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader)
    class_weights = class_weights.to(args.device)
    
    # Define combined loss function
    criterion = CombinedLoss(
        alpha=class_weights,
        gamma=2.0,
        iou_weight=0.3  # Can be adjusted based on performance
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    # Initialize best validation metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Train for specified number of epochs
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_best_loss.pth'))
            print(f"Saved best loss model at epoch {epoch+1}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_best_acc.pth'))
            print(f"Saved best accuracy model at epoch {epoch+1}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, os.path.join(model_dir, f'{model_name}_latest.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'{model_name}_training_curves.png'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
    }
    
    torch.save(history, os.path.join(model_dir, f'{model_name}_history.pth'))
    
    print(f"\n=== {model_name} Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_loss, best_val_acc

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get custom data loaders with our collate function
    train_dataset, val_dataset = get_dataset_splits(
        data_dir=args.data_dir,
        window_size=args.window_size,
        window_step=args.window_step,
        sequence_length=args.max_seq_length,
        cross_val_mode=args.cross_val,
        augment=True,
        normalize=True
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Initialize metrics storage
    results = {}
    
    # Train CNN3D
    if args.model_type in ['cnn3d', 'all']:
        model = CNN3D(
            max_seq_length=args.max_seq_length,
            in_channels=1,
            out_channels=2  # Binary segmentation
        ).to(device)
        
        loss, acc = train_model('cnn3d', model, train_loader, val_loader, args)
        results['cnn3d'] = {'loss': loss, 'acc': acc}
    
    # Train Vision Transformer
    if args.model_type in ['vit', 'all']:
        model = VisionTransformer(
            img_size=args.window_size,
            patch_size=16,  # 16x16 patches as specified
            in_channels=1,
            max_seq_length=args.max_seq_length,
            num_classes=2,  # Binary segmentation
            embed_dim=256,
            depth=6,
            num_heads=8
        ).to(device)
        
        loss, acc = train_model('vit', model, train_loader, val_loader, args)
        results['vit'] = {'loss': loss, 'acc': acc}
    
    # Train MambaCNN
    if args.model_type in ['mamba', 'all']:
        model = MambaCNN(
            img_size=args.window_size,
            in_channels=1,
            max_seq_length=args.max_seq_length,
            num_classes=2,  # Binary segmentation
            base_channels=64,
            d_state=16,
            depth=2
        ).to(device)
        
        loss, acc = train_model('mamba', model, train_loader, val_loader, args)
        results['mamba'] = {'loss': loss, 'acc': acc}
    
    # Print and save results summary
    print("\n=== Training Results ===")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Best val loss: {metrics['loss']:.4f}")
        print(f"  Best val accuracy: {metrics['acc']:.2f}%")
    
    # Save results summary
    results_file = os.path.join(args.output_dir, 'training_results.txt')
    with open(results_file, 'w') as f:
        f.write("=== Training Results ===\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Best val loss: {metrics['loss']:.4f}\n")
            f.write(f"  Best val accuracy: {metrics['acc']:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")

# Function to get train and validation datasets
def get_dataset_splits(data_dir, window_size, window_step, sequence_length, cross_val_mode, augment, normalize):
    """Helper function to get dataset splits without creating DataLoaders"""
    
    # Create training dataset with frame filtering
    train_dataset = FilteredTemporalWindowDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=sequence_length,
        cross_val_mode=cross_val_mode,
        train=True,
        augment=augment,
        normalize=normalize
    )
    
    # Create validation dataset with frame filtering
    val_dataset = FilteredTemporalWindowDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=sequence_length,
        cross_val_mode=cross_val_mode,
        train=False,
        augment=False,  # No augmentation for validation
        normalize=normalize
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    main() 
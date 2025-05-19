import os
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import TemporalWindowDataset
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN
from train import train_model, custom_collate_fn, calculate_class_weights
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LimitedTemporalWindowDataset(TemporalWindowDataset):
    """
    Dataset that loads only the first 25 frames (0-24) from the original dataset.
    """
    def _load_data(self):
        """
        Override the _load_data method to only load the first 25 frames.
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
            
        # Limit to only frames 0-24
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train final models using best parameters from ablation study')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results/final_models', help='Path to output directory')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['cnn3d', 'vit', 'mamba', 'all'], 
                        help='Model type to train')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda or cpu)')
    
    return parser.parse_args()

def evaluate_miou(model, data_loader, device):
    """
    Evaluate Mean IoU on a dataset
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to use
        
    Returns:
        Mean IoU score
    """
    model.eval()
    
    # Initialize accumulators for each class
    num_classes = 2  # Binary segmentation
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Calculating MIoU'):
            # Process batch at once
            data = data.to(device)
            target = target.to(device)
            
            # Get predictions (B, C, H, W) -> (B, H, W)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            # Process each class (vectorized)
            for cls in range(num_classes):
                # Create binary masks
                pred_mask = (preds == cls)
                target_mask = (target == cls)
                
                # Update intersection and union (vectorized operations)
                intersection[cls] += torch.logical_and(pred_mask, target_mask).sum().item()
                union[cls] += torch.logical_or(pred_mask, target_mask).sum().item()
    
    # Calculate IoU for each class, avoiding division by zero
    iou = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        if union[cls] > 0:
            iou[cls] = intersection[cls] / union[cls]
        else:
            iou[cls] = 1.0  # If no predictions or targets, set to 1
    
    # Calculate mean IoU
    miou = iou.mean().item()
    
    return miou

def train_final_model(model_type, args):
    """Train a model with the best parameters from the ablation study"""
    print(f"\n=== Training final {model_type} model ===")
    
    # Best parameters from ablation study
    if model_type == 'cnn3d':
        window_size = 256
        window_step = 64
        max_seq_length = 10
    elif model_type == 'vit':
        window_size = 256
        window_step = 32
        max_seq_length = 15
    elif model_type == 'mamba':
        window_size = 256
        window_step = 64
        max_seq_length = 5
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create output directory for this model
    model_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create datasets with best configuration, using limited dataset
    train_dataset = LimitedTemporalWindowDataset(
        data_dir=args.data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=max_seq_length,
        cross_val_mode='left',
        train=True,
        augment=True,
        normalize=True
    )
    
    val_dataset = LimitedTemporalWindowDataset(
        data_dir=args.data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=max_seq_length,
        cross_val_mode='left',
        train=False,
        augment=False,
        normalize=True
    )
    
    # Create data loaders
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
    
    # Initialize model
    if model_type == 'cnn3d':
        model = CNN3D(
            max_seq_length=max_seq_length,
            in_channels=1,
            out_channels=2
        ).to(args.device)
    elif model_type == 'vit':
        model = VisionTransformer(
            img_size=window_size,
            patch_size=16,
            in_channels=1,
            max_seq_length=max_seq_length,
            num_classes=2,
            embed_dim=256,
            depth=6,
            num_heads=8
        ).to(args.device)
    elif model_type == 'mamba':
        model = MambaCNN(
            img_size=window_size,
            in_channels=1,
            max_seq_length=max_seq_length,
            num_classes=2,
            base_channels=64,
            d_state=16,
            depth=2
        ).to(args.device)
    
    # Train model
    loss, acc = train_model(model_type, model, train_loader, val_loader, args)
    
    # Calculate MIoU on validation set
    val_miou = evaluate_miou(model, val_loader, args.device)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_type}_final.pth"))
    
    # Save model config
    config = {
        'model_type': model_type,
        'window_size': window_size,
        'window_step': window_step,
        'max_seq_length': max_seq_length,
        'val_loss': loss,
        'val_acc': acc,
        'val_miou': val_miou
    }
    
    config_df = pd.DataFrame([config])
    config_df.to_csv(os.path.join(model_dir, f"{model_type}_config.csv"), index=False)
    
    print(f"\nFinal {model_type} model results:")
    print(f"Window size: {window_size}")
    print(f"Window step: {window_step}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.2f}%")
    print(f"Validation MIoU: {val_miou:.4f}")
    
    return loss, acc, val_miou

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which models to train
    if args.model_type == 'all':
        model_types = ['cnn3d', 'vit', 'mamba']
    else:
        model_types = [args.model_type]
    
    # Results file path
    results_file = os.path.join(args.output_dir, 'final_results.csv')
    
    # Train each model
    results = []
    for model_type in model_types:
        loss, acc, miou = train_final_model(model_type, args)
        
        results.append({
            'model': model_type,
            'val_loss': loss,
            'val_acc': acc,
            'val_miou': miou
        })
    
    # Save and display final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print("\n=== Final Training Complete ===")
    print("\nResults:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 
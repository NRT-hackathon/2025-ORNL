import os
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders, CrossValidationMode, TemporalWindowDataset
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN
from train import train_model, custom_collate_fn, calculate_class_weights, CombinedLoss, FocalLoss
import cv2
import torch.optim as optim
import torch.nn.functional as F

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
        frame_files = frame_files[:25]
        label_files = label_files[:25]
        
        print(f"Found {len(frame_files)} frame files and {len(label_files)} label files")
        print(f"Using only frames 0-24 for training/validation")
        
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
    parser = argparse.ArgumentParser(description='Run ablation studies and cross-validation')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='Path to output directory')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['cnn3d', 'vit', 'mamba', 'all'], 
                        help='Model type to train')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--ablation_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--cross_val_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda or cpu)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints if available')
    
    return parser.parse_args()

def get_ablation_configs():
    """Return a list of configurations to test in the ablation study"""
    configs = []
    
    # Window sizes to test
    window_sizes = [64, 128, 256]
    
    # Window steps to test (as fraction of window size)
    step_ratios = [0.125, 0.25, 0.5]
    
    # Sequence lengths to test
    seq_lengths = [5]#, 10, 15]
    
    # Generate all combinations
    for window_size in window_sizes:
        for ratio in step_ratios:
            window_step = int(window_size * ratio)
            for seq_len in seq_lengths:
                configs.append({
                    'window_size': window_size,
                    'window_step': window_step,
                    'max_seq_length': seq_len
                })
    
    return configs

def calculate_miou(preds, targets, num_classes=2):
    """
    Calculate Mean Intersection over Union (MIoU) for segmentation.
    
    Args:
        preds: Model predictions (B, C, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes
        
    Returns:
        MIoU score
    """
    # Convert predictions to class indices
    preds = torch.argmax(preds, dim=1)  # (B, H, W)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    
    # Update confusion matrix
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate IoU for each class
    iou = torch.zeros(num_classes, device=preds.device)
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        
        # Avoid division by zero
        if tp + fp + fn == 0:
            iou[c] = 1.0
        else:
            iou[c] = tp / (tp + fp + fn)
    
    # Calculate mean IoU
    miou = iou.mean()
    
    return miou.item()

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
    
    # Return only the positive class IoU if needed
    # pos_iou = iou[1].item()
    
    return miou

def load_existing_results(results_file):
    """Load existing results from a CSV file if it exists"""
    if os.path.exists(results_file):
        return pd.read_csv(results_file).to_dict('records')
    return []

def save_results(results, results_file):
    """Save results to a CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    return df

def config_to_string(config):
    """Convert a configuration dictionary to a string representation"""
    return f"ws{config['window_size']}_wst{config['window_step']}_seq{config['max_seq_length']}"

def ensure_config_types(config):
    """Ensure configuration values have the correct types"""
    if 'window_size' in config:
        config['window_size'] = int(config['window_size'])
    if 'window_step' in config:
        config['window_step'] = int(config['window_step'])
    if 'max_seq_length' in config:
        config['max_seq_length'] = int(config['max_seq_length'])
    return config

def run_ablation_study(model_type, configs, args):
    """Run ablation study for a specific model type"""
    print(f"\n=== Running ablation study for {model_type} ===")
    
    # Create output directory for this model type
    model_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Results file path
    results_file = os.path.join(model_dir, f"{model_type}_ablation_results.csv")
    
    # Load existing results if resuming
    results = []
    if args.resume and os.path.exists(results_file):
        results = load_existing_results(results_file)
        print(f"Loaded {len(results)} existing results from {results_file}")
    
    # Keep track of completed configurations
    completed_configs = {config_to_string({
        'window_size': result['window_size'],
        'window_step': result['window_step'],
        'max_seq_length': result['max_seq_length']
    }) for result in results}
    
    # Run each configuration
    for config in tqdm(configs, desc=f"Testing {model_type} configurations"):
        # Check if this configuration has already been completed
        config_str = config_to_string(config)
        model_path = os.path.join(model_dir, f"{model_type}_{config_str}.pth")
        
        if args.resume and config_str in completed_configs:
            print(f"Skipping already completed configuration: {config_str}")
            continue
        
        print(f"Running configuration: {config_str}")
        
        # Create dataset with current configuration
        train_dataset = LimitedTemporalWindowDataset(
            data_dir=args.data_dir,
            window_size=config['window_size'],
            window_step=config['window_step'],
            sequence_length=config['max_seq_length'],
            cross_val_mode='top',  # Use top for initial ablation
            train=True,
            augment=True,
            normalize=True
        )
        
        val_dataset = LimitedTemporalWindowDataset(
            data_dir=args.data_dir,
            window_size=config['window_size'],
            window_step=config['window_step'],
            sequence_length=config['max_seq_length'],
            cross_val_mode='top',
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
                max_seq_length=config['max_seq_length'],
                in_channels=1,
                out_channels=2
            ).to(args.device)
        elif model_type == 'vit':
            model = VisionTransformer(
                img_size=config['window_size'],
                patch_size=16,
                in_channels=1,
                max_seq_length=config['max_seq_length'],
                num_classes=2,
                embed_dim=256,
                depth=6,
                num_heads=8
            ).to(args.device)
        elif model_type == 'mamba':
            model = MambaCNN(
                img_size=config['window_size'],
                in_channels=1,
                max_seq_length=config['max_seq_length'],
                num_classes=2,
                base_channels=64,
                d_state=16,
                depth=2
            ).to(args.device)
        
        # Set epochs for ablation study
        args.epochs = args.ablation_epochs
        
        # Train model
        loss, acc = train_model(model_type, model, train_loader, val_loader, args)
        
        # Calculate MIoU on validation set
        val_miou = evaluate_miou(model, val_loader, args.device)
        
        # Save results
        new_result = {
            'window_size': config['window_size'],
            'window_step': config['window_step'],
            'max_seq_length': config['max_seq_length'],
            'val_loss': loss,
            'val_acc': acc,
            'val_miou': val_miou
        }
        results.append(new_result)
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Update results file after each configuration to enable resuming
        save_results(results, results_file)
        
        # Mark this configuration as completed
        completed_configs.add(config_str)
    
    # Find best configuration based on MIoU
    df = pd.DataFrame(results)
    best_config_idx = df['val_miou'].idxmax()
    best_config = df.loc[best_config_idx].to_dict()
    
    # Ensure the best config has integer values
    best_config = ensure_config_types(best_config)
    
    print(f"\nBest configuration for {model_type}:")
    print(f"Window size: {best_config['window_size']}")
    print(f"Window step: {best_config['window_step']}")
    print(f"Max sequence length: {best_config['max_seq_length']}")
    print(f"Validation MIoU: {best_config['val_miou']:.4f}")
    
    return best_config

def run_cross_validation(model_type, best_config, args):
    """Run 4-fold cross-validation using the best configuration"""
    print(f"\n=== Running 4-fold cross-validation for {model_type} ===")
    
    # Ensure best_config values are integers
    best_config = ensure_config_types(best_config)
    
    # Create output directory for cross-validation results
    cv_dir = os.path.join(args.output_dir, model_type, 'cross_validation')
    os.makedirs(cv_dir, exist_ok=True)
    
    # Results file path
    results_file = os.path.join(cv_dir, f"{model_type}_cv_results.csv")
    
    # Load existing results if resuming
    results = []
    if args.resume and os.path.exists(results_file):
        results = load_existing_results(results_file)
        print(f"Loaded {len(results)} existing cross-validation results from {results_file}")
    
    # Keep track of completed CV modes
    completed_modes = {result['cv_mode'] for result in results}
    
    # Cross-validation modes
    cv_modes = ['top', 'left', 'right', 'bottom']
    
    for cv_mode in cv_modes:
        # Check if this CV mode has already been completed
        model_path = os.path.join(cv_dir, f"{model_type}_{cv_mode}.pth")
        
        if args.resume and cv_mode in completed_modes:
            print(f"Skipping already completed cross-validation mode: {cv_mode}")
            continue
            
        print(f"\nRunning cross-validation with {cv_mode} validation set")
        
        # Create dataset with current configuration
        train_dataset = LimitedTemporalWindowDataset(
            data_dir=args.data_dir,
            window_size=best_config['window_size'],
            window_step=best_config['window_step'],
            sequence_length=best_config['max_seq_length'],
            cross_val_mode=cv_mode,
            train=True,
            augment=True,
            normalize=True
        )
        
        val_dataset = LimitedTemporalWindowDataset(
            data_dir=args.data_dir,
            window_size=best_config['window_size'],
            window_step=best_config['window_step'],
            sequence_length=best_config['max_seq_length'],
            cross_val_mode=cv_mode,
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
                max_seq_length=best_config['max_seq_length'],
                in_channels=1,
                out_channels=2
            ).to(args.device)
        elif model_type == 'vit':
            model = VisionTransformer(
                img_size=best_config['window_size'],
                patch_size=16,
                in_channels=1,
                max_seq_length=best_config['max_seq_length'],
                num_classes=2,
                embed_dim=256,
                depth=6,
                num_heads=8
            ).to(args.device)
        elif model_type == 'mamba':
            model = MambaCNN(
                img_size=best_config['window_size'],
                in_channels=1,
                max_seq_length=best_config['max_seq_length'],
                num_classes=2,
                base_channels=64,
                d_state=16,
                depth=2
            ).to(args.device)
        
        # Set epochs for cross validation
        args.epochs = args.cross_val_epochs
        
        # Train model
        loss, acc = train_model(model_type, model, train_loader, val_loader, args)
        
        # Calculate MIoU on validation set
        val_miou = evaluate_miou(model, val_loader, args.device)
        
        # Save results
        new_result = {
            'cv_mode': cv_mode,
            'val_loss': loss,
            'val_acc': acc,
            'val_miou': val_miou
        }
        results.append(new_result)
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Update results file after each fold to enable resuming
        save_results(results, results_file)
        
        # Mark this CV mode as completed
        completed_modes.add(cv_mode)
    
    # Calculate and print average metrics
    df = pd.DataFrame(results)
    avg_loss = df['val_loss'].mean()
    avg_acc = df['val_acc'].mean()
    avg_miou = df['val_miou'].mean()
    std_miou = df['val_miou'].std()
    
    print(f"\nCross-validation results for {model_type}:")
    print(f"Average validation loss: {avg_loss:.4f}")
    print(f"Average validation accuracy: {avg_acc:.2f}%")
    print(f"Average validation MIoU: {avg_miou:.4f} ± {std_miou:.4f}")
    
    return avg_loss, avg_miou, std_miou

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get ablation configurations
    configs = get_ablation_configs()
    
    # Determine which models to evaluate
    if args.model_type == 'all':
        model_types = ['cnn3d', 'vit', 'mamba']
    else:
        model_types = [args.model_type]
    
    # Final results file path
    final_results_file = os.path.join(args.output_dir, 'ablation.csv')
    
    # Load existing final results if resuming
    final_results = []
    if args.resume and os.path.exists(final_results_file):
        final_results = load_existing_results(final_results_file)
        print(f"Loaded {len(final_results)} existing final results from {final_results_file}")
        
        # Ensure all loaded configurations have integer values
        for result in final_results:
            ensure_config_types(result)
    
    # Keep track of completed models
    completed_models = {result['model'] for result in final_results}
    
    # Run ablation study and cross-validation for each model
    for model_type in model_types:
        if args.resume and model_type in completed_models:
            print(f"Skipping already completed model: {model_type}")
            continue
        
        # Run ablation study
        best_config = run_ablation_study(model_type, configs, args)
        
        # Run cross-validation
        avg_loss, avg_miou, std_miou = run_cross_validation(model_type, best_config, args)
        
        # Store results
        new_result = {
            'model': model_type,
            'window_size': best_config['window_size'],
            'window_step': best_config['window_step'],
            'max_seq_length': best_config['max_seq_length'],
            'avg_val_loss': avg_loss,
            'avg_val_miou': avg_miou,
            'std_val_miou': std_miou
        }
        final_results.append(new_result)
        
        # Update final results file after each model to enable resuming
        save_results(final_results, final_results_file)
        
        # Mark this model as completed
        completed_models.add(model_type)
    
    print("\n=== Ablation Study Complete ===")
    print("\nFinal Results:")
    df = pd.DataFrame(final_results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 
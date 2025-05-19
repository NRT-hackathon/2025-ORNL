import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from dataset import TemporalWindowDataset, CrossValidationMode
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN

def load_model(model_type, hyperparams, model_path, device):
    """Load a trained model with appropriate hyperparameters"""
    max_seq_length = hyperparams['max_seq_length']
    window_size = hyperparams['window_size']
    
    if model_type == 'cnn3d':
        model = CNN3D(max_seq_length=max_seq_length, in_channels=1, out_channels=2)
    elif model_type == 'mamba':
        model = MambaCNN(
            img_size=window_size,
            in_channels=1,
            max_seq_length=max_seq_length,
            num_classes=2,
            base_channels=64,
            d_state=16,
            depth=2
        )
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
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

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
        print(f"Using only frames 0-24 for evaluation")
        
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

class FixedSequenceLengthDataset(LimitedTemporalWindowDataset):
    """Dataset that only creates sequences of a specific length"""
    
    def __init__(self, data_dir, window_size, window_step, max_seq_length, 
                 fixed_seq_length, cross_val_mode='top', train=False, 
                 augment=False, normalize=True):
        """
        Initialize dataset with a fixed sequence length
        
        Args:
            data_dir: Directory containing frames and labels
            window_size: Size of spatial window
            window_step: Step size for sliding window
            max_seq_length: Maximum sequence length the model supports
            fixed_seq_length: The exact sequence length to use (1 to max_seq_length)
            cross_val_mode: Cross-validation mode
            train: Whether this is training or validation set
            augment: Whether to apply augmentation
            normalize: Whether to normalize data
        """
        self.fixed_seq_length = min(fixed_seq_length, max_seq_length)
        super().__init__(data_dir, window_size, window_step, max_seq_length,
                        cross_val_mode, train, augment, normalize)
    
    def _create_sequences(self):
        """Create temporal sequences with only the fixed sequence length"""
        sequences = []
        
        num_frames = len(self.frames)
        positions_to_use = self.positions_to_use
        
        # For each position
        for pos_idx, (x, y) in enumerate(positions_to_use):
            # Only use the fixed sequence length
            seq_len = self.fixed_seq_length
            
            # For each possible starting frame that allows this sequence length
            for start_frame in range(num_frames - seq_len + 1):
                end_frame = start_frame + seq_len - 1
                sequences.append({
                    "pos_idx": pos_idx,
                    "x": x,
                    "y": y,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "seq_len": seq_len
                })
        
        return sequences

def evaluate_miou(model, data_loader, device):
    """Evaluate Mean IoU on a dataset"""
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

def custom_collate_fn(batch):
    """
    Custom collate function to handle varied sequence lengths.
    
    This function ensures that all sequences in a batch have the same length
    by padding shorter sequences with zeros. It is needed when batching
    sequences of different lengths.
    
    Args:
        batch: List of (sequence, target) tuples
        
    Returns:
        Batched sequences and targets
    """
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Convert to tensors
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    
    return sequences, targets

def evaluate_temporal_performance(model_type, data_dir, models_dir, output_dir, device):
    """
    Evaluate model performance across different temporal window sizes
    
    Args:
        model_type: Type of model to evaluate ('cnn3d', 'vit', 'mamba')
        data_dir: Directory containing the data
        models_dir: Directory containing the trained models
        output_dir: Directory to save results
        device: Device to use for evaluation
    """
    print(f"Evaluating temporal performance for {model_type} model")
    
    # Load model configuration
    config_path = os.path.join(models_dir, model_type, f"{model_type}_config.csv")
    config_df = pd.read_csv(config_path)
    hyperparams = config_df.iloc[0].to_dict()
    
    # Model path
    model_path = os.path.join(models_dir, model_type, f"{model_type}_final.pth")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_type, hyperparams, model_path, device)
    
    # Get parameters
    window_size = int(hyperparams['window_size'])
    window_step = int(hyperparams['window_step'])
    max_seq_length = int(hyperparams['max_seq_length'])
    
    print(f"Model parameters: window_size={window_size}, window_step={window_step}, max_seq_length={max_seq_length}")
    
    # Evaluate for each sequence length from 1 to max_seq_length
    results = []
    
    for seq_length in range(1, max_seq_length + 1):
        print(f"Evaluating with sequence length {seq_length}...")
        
        # Create dataset with this sequence length
        val_dataset = FixedSequenceLengthDataset(
            data_dir=data_dir,
            window_size=window_size,
            window_step=window_step,
            max_seq_length=max_seq_length,
            fixed_seq_length=seq_length,
            cross_val_mode='top',
            train=False,
            augment=False,
            normalize=True
        )
        
        # Create data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Evaluate model
        miou = evaluate_miou(model, val_loader, device)
        
        # Record results
        results.append({
            'model': model_type,
            'sequence_length': seq_length,
            'miou': miou
        })
        
        print(f"Sequence length {seq_length} - MIoU: {miou:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, f"{model_type}_temporal_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['sequence_length'], results_df['miou'], marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean IoU')
    plt.title(f'{model_type.upper()} Performance vs. Temporal Window Size')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{model_type}_temporal_performance.png")
    plt.savefig(plot_path)
    plt.close()
    
    return results_df

def compare_models(results_dict, output_dir):
    """
    Create a comparison plot for all models
    
    Args:
        results_dict: Dictionary mapping model_type to results DataFrame
        output_dir: Directory to save results
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    for model_type, df in results_dict.items():
        plt.plot(df['sequence_length'], df['miou'], marker='o', label=model_type.upper())
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean IoU')
    plt.title('Model Performance vs. Temporal Window Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "model_comparison_temporal.png")
    plt.savefig(plot_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance across temporal window sizes')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='results/final_models', help='Path to models directory')
    parser.add_argument('--output_dir', type=str, default='results/temporal_analysis', help='Path to output directory')
    parser.add_argument('--model_type', type=str, default='all', choices=['cnn3d', 'vit', 'mamba', 'all'], help='Model type to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Determine which models to evaluate
    if args.model_type == 'all':
        model_types = ['cnn3d', 'vit', 'mamba']
    else:
        model_types = [args.model_type]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each model
    results_dict = {}
    
    for model_type in model_types:
        results_df = evaluate_temporal_performance(
            model_type, 
            args.data_dir, 
            args.models_dir, 
            args.output_dir,
            device
        )
        results_dict[model_type] = results_df
    
    # Create comparison plot if multiple models
    if len(model_types) > 1:
        compare_models(results_dict, args.output_dir)

if __name__ == "__main__":
    main() 
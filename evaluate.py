import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders, CrossValidationMode, TemporalWindowDataset
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from torchvision.utils import make_grid, save_image
import seaborn as sns
import pandas as pd
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a dataset class that only loads frames 0-49
class LimitedTemporalWindowDataset(TemporalWindowDataset):
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
        frame_files = frame_files[25:50]
        label_files = label_files[25:50]
        
        print(f"Found {len(frame_files)} frame files and {len(label_files)} label files")
        print(f"Using only frames 0-49 for evaluation")
        
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
    parser = argparse.ArgumentParser(description='Evaluate defect detection models')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='results/final_models', help='Path to trained models directory')
    parser.add_argument('--results_dir', type=str, default='results/evaluation', help='Path to save evaluation results')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['cnn3d', 'vit', 'mamba', 'all'], 
                        help='Model type to evaluate')
    parser.add_argument('--model_selection', type=str, default='best_acc', 
                        choices=['best_acc', 'best_loss', 'latest'],
                        help='Which model checkpoint to use')
    
    # Ablation file for hyperparameters
    parser.add_argument('--ablation_file', type=str, default='results/ablation/ablation.csv',
                        help='CSV file with model hyperparameters')
    
    # Final results file
    parser.add_argument('--final_results_file', type=str, default='results/final_models/final_results.csv',
                        help='CSV file with final model results')
    
    # Dataset parameters (these will be overridden by ablation file values)
    parser.add_argument('--window_size', type=int, default=256, help='Window size for patches')
    parser.add_argument('--window_step', type=int, default=64, help='Step size for sliding window')
    parser.add_argument('--max_seq_length', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--cross_val', type=str, default='top', 
                        choices=['top', 'left', 'right', 'bottom', 'none'],
                        help='Cross-validation mode')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of example images to save')
    
    return parser.parse_args()

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels (flattened)
        y_pred: Predicted labels (flattened)
        
    Returns:
        Dictionary of metrics
    """
    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Calculate IoU (Jaccard index)
    iou = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'iou': iou * 100
    }

def evaluate_model(model, data_loader, device, num_samples=10):
    """
    Evaluate a model on the validation set.
    
    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to use
        num_samples: Number of sample images to save
        
    Returns:
        Dictionary of metrics and sample predictions
    """
    model.eval()
    
    # Initialize accumulators for metrics
    total_correct = 0
    total_pixels = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    sample_inputs = []
    sample_targets = []
    sample_predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc='Evaluating')):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            
            # Calculate metrics for this batch
            correct = (predictions == target).sum().item()
            total_correct += correct
            total_pixels += target.numel()
            
            # Calculate true positives, false positives, false negatives
            true_positives += ((predictions == 1) & (target == 1)).sum().item()
            false_positives += ((predictions == 1) & (target == 0)).sum().item()
            false_negatives += ((predictions == 0) & (target == 1)).sum().item()
            
            # Collect samples if needed
            if len(sample_inputs) < num_samples:
                # Take first sample from batch
                sample_inputs.append(data[0].cpu())
                sample_targets.append(target[0].cpu())
                sample_predictions.append(predictions[0].cpu())
            
            # Clear memory
            del data, target, outputs, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Calculate final metrics
    accuracy = total_correct / total_pixels
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'iou': iou * 100
    }
    
    # Prepare samples
    samples = {
        'inputs': sample_inputs[:num_samples],
        'targets': sample_targets[:num_samples],
        'predictions': sample_predictions[:num_samples]
    }
    
    return metrics, samples

def save_sample_predictions(samples, model_name, results_dir):
    """
    Save sample predictions as images.
    
    Args:
        samples: Dictionary of sample inputs, targets, and predictions
        model_name: Name of the model
        results_dir: Directory to save results
    """
    os.makedirs(os.path.join(results_dir, 'samples'), exist_ok=True)
    
    for i, (input_seq, target, prediction) in enumerate(zip(
        samples['inputs'], samples['targets'], samples['predictions']
    )):
        # Create a figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot input (last frame)
        axes[0].imshow(input_seq[-1].numpy(), cmap='gray')
        axes[0].set_title('Input (Last Frame)')
        axes[0].axis('off')
        
        # Plot target
        axes[1].imshow(target.numpy(), cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(prediction.numpy(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'samples', f'{model_name}_sample_{i+1}.png'))
        plt.close()

def initialize_model(model_type, hyperparams):
    """
    Initialize a model based on the model type.
    
    Args:
        model_type: Type of model to initialize
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Initialized model
    """
    max_seq_length = hyperparams['max_seq_length']
    window_size = hyperparams['window_size']
    
    if model_type == 'cnn3d':
        model = CNN3D(
            max_seq_length=max_seq_length,
            in_channels=1,
            out_channels=2  # Binary segmentation
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def load_model(model, model_type, args):
    """
    Load trained model weights.
    
    Args:
        model: Model to load weights into
        model_type: Type of model
        args: Command-line arguments
        
    Returns:
        Model with loaded weights
    """
    model_dir = os.path.join(args.models_dir, model_type)
    
    # Try to find the model file
    if os.path.exists(os.path.join(model_dir, f'{model_type}.pth')):
        checkpoint_path = os.path.join(model_dir, f'{model_type}.pth')
    else:
        checkpoint_path = os.path.join(model_dir, f'{model_type}_{args.model_selection}.pth')
    
    print(f"Loading model from {checkpoint_path}")
    
    try:
        # Try loading as a full checkpoint first
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If not a dict with model_state_dict, assume it's just the state dict
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return model

def plot_comparison_chart(all_metrics, results_dir):
    """
    Plot comparison chart of metrics across models.
    
    Args:
        all_metrics: Dictionary of metrics for each model
        results_dir: Directory to save results
    """
    # Convert to DataFrame for easier plotting
    data = []
    for model_name, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title('Model Comparison')
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    plt.close()
    
    # Also create a table
    pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=pivot_df.round(2).values,
                     rowLabels=pivot_df.index,
                     colLabels=pivot_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.savefig(os.path.join(results_dir, 'metrics_table.png'), bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV file
    final_results_df = pd.DataFrame()
    for model_name, metrics in all_metrics.items():
        row = {'model': model_name}
        row.update(metrics)
        final_results_df = pd.concat([final_results_df, pd.DataFrame([row])], ignore_index=True)
    
    final_results_df.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)

def load_hyperparameters(ablation_file, model_type):
    """
    Load model hyperparameters from ablation CSV file
    
    Args:
        ablation_file: Path to ablation CSV file
        model_type: Type of model to get hyperparameters for
        
    Returns:
        Dictionary of hyperparameters
    """
    try:
        ablation_df = pd.read_csv(ablation_file)
        model_params = ablation_df[ablation_df['model'] == model_type].iloc[0]
        
        hyperparams = {
            'window_size': int(model_params['window_size']),
            'window_step': int(model_params['window_step']),
            'max_seq_length': int(model_params['max_seq_length'])
        }
        
        return hyperparams
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        print("Using default hyperparameters")
        return {
            'window_size': 256,
            'window_step': 64,
            'max_seq_length': 10
        }

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine which models to evaluate
    if args.model_type == 'all':
        model_types = ['cnn3d', 'vit', 'mamba']
    else:
        model_types = [args.model_type]
    
    # Dictionary to store metrics for all models
    all_metrics = {}
    
    # Evaluate each model
    for model_type in model_types:
        print(f"\n=== Evaluating {model_type} ===")
        
        # Load hyperparameters for this model from ablation file
        hyperparams = load_hyperparameters(args.ablation_file, model_type)
        print(f"Using hyperparameters: {hyperparams}")
        
        # Get data loader (validation only) using the LimitedTemporalWindowDataset with model-specific parameters
        val_dataset = LimitedTemporalWindowDataset(
            data_dir=args.data_dir,
            window_size=hyperparams['window_size'],
            window_step=hyperparams['window_step'],
            sequence_length=hyperparams['max_seq_length'],
            cross_val_mode=args.cross_val,
            train=False,
            augment=False,
            normalize=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Initialize model with appropriate hyperparameters
        model = initialize_model(model_type, hyperparams)
        
        # Load trained weights
        try:
            model = load_model(model, model_type, args)
            model = model.to(device)
        except FileNotFoundError:
            print(f"Model file not found. Skipping {model_type}.")
            continue
        
        # Evaluate model
        metrics, samples = evaluate_model(model, val_loader, device, args.num_samples)
        print('eval done')
        
        # Store metrics
        all_metrics[model_type] = metrics
        
        # Save sample predictions
        save_sample_predictions(samples, model_type, args.results_dir)
        
        # Print metrics
        print(f"Metrics for {model_type}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.2f}%")
        
        # Save metrics to file
        with open(os.path.join(args.results_dir, f'{model_type}_metrics.txt'), 'w') as f:
            f.write(f"Metrics for {model_type}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.2f}%\n")
    
    # Plot comparison chart if multiple models were evaluated
    if len(all_metrics) > 1:
        plot_comparison_chart(all_metrics, args.results_dir)
        print(f"\nComparison chart saved to {os.path.join(args.results_dir, 'model_comparison.png')}")
        print(f"Metrics table saved to {os.path.join(args.results_dir, 'metrics_table.png')}")
        print(f"Evaluation results saved to {os.path.join(args.results_dir, 'evaluation_results.csv')}")

if __name__ == "__main__":
    main() 
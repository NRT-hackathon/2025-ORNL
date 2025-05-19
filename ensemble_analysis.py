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

class EnsembleModel(torch.nn.Module):
    """
    Weighted ensemble of multiple models
    
    This class combines predictions from multiple models using weighted averaging.
    """
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble
        
        Args:
            models: Dictionary mapping model names to model instances
            weights: Dictionary mapping model names to weights (will be normalized)
        """
        super().__init__()
        self.models = models
        
        # If weights not provided, use equal weighting
        if weights is None:
            weights = {name: 1.0 for name in models.keys()}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {name: weight / total for name, weight in weights.items()}
        
        print("Ensemble weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
    
    def forward(self, x):
        """
        Forward pass through the ensemble
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model predictions
        """
        # Get predictions from each model
        outputs = {}
        for name, model in self.models.items():
            outputs[name] = model(x)
        
        # Weighted average of predictions
        ensemble_output = None
        for name, output in outputs.items():
            if ensemble_output is None:
                ensemble_output = self.weights[name] * output
            else:
                ensemble_output += self.weights[name] * output
        
        return ensemble_output

    def eval(self):
        """Set all models to evaluation mode"""
        for model in self.models.values():
            model.eval()
        return self

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
        frame_files = frame_files[:25]
        label_files = label_files[:25]
        
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

def determine_common_parameters(model_configs):
    """
    Determine common parameters across models for ensemble compatibility
    
    Args:
        model_configs: Dictionary mapping model types to hyperparameters
        
    Returns:
        Dictionary of common parameters
    """
    # Find smallest window size and step
    window_sizes = [int(config['window_size']) for config in model_configs.values()]
    window_steps = [int(config['window_step']) for config in model_configs.values()]
    max_seq_lengths = [int(config['max_seq_length']) for config in model_configs.values()]
    
    # Use the smallest max sequence length
    min_max_seq_length = min(max_seq_lengths)
    
    # For window size and step, use the most common value
    from collections import Counter
    window_size = Counter(window_sizes).most_common(1)[0][0]
    window_step = Counter(window_steps).most_common(1)[0][0]
    
    return {
        'window_size': window_size,
        'window_step': window_step,
        'max_seq_length': min_max_seq_length
    }

def create_ensemble(model_types, models_dir, device, weights=None):
    """
    Create an ensemble model from individual models
    
    Args:
        model_types: List of model types to include in the ensemble
        models_dir: Directory containing trained models
        device: Device to use for models
        weights: Optional dictionary mapping model types to weights
        
    Returns:
        Ensemble model, common parameters for the ensemble
    """
    # Load model configurations
    model_configs = {}
    models = {}
    
    for model_type in model_types:
        config_path = os.path.join(models_dir, model_type, f"{model_type}_config.csv")
        config_df = pd.read_csv(config_path)
        hyperparams = config_df.iloc[0].to_dict()
        model_configs[model_type] = hyperparams
        
        # Load model
        model_path = os.path.join(models_dir, model_type, f"{model_type}_final.pth")
        models[model_type] = load_model(model_type, hyperparams, model_path, device)
    
    # Determine common parameters for the ensemble
    common_params = determine_common_parameters(model_configs)
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights)
    
    return ensemble, common_params

def generate_weight_schemes(model_types):
    """
    Generate comprehensive permutations of weighting schemes for ensemble models.
    Weights vary from 0 to 1.0 in increments of 0.2.
    
    Args:
        model_types: List of model types to include
        
    Returns:
        Dictionary of weighting schemes
    """
    schemes = {
        'equal': {model: 1.0 for model in model_types},  # Equal weights
    }
    
    # Create all permutations of weights with 0.2 increments (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    weight_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    if len(model_types) == 3:  # Specifically for 3 models (cnn3d, vit, mamba)
        # Generate all combinations
        for w1 in weight_values:
            for w2 in weight_values:
                for w3 in weight_values:
                    # Skip if all weights are zero
                    if w1 == 0 and w2 == 0 and w3 == 0:
                        continue
                        
                    # Create name based on weights
                    name = f"{model_types[0]}_{w1:.1f}_{model_types[1]}_{w2:.1f}_{model_types[2]}_{w3:.1f}"
                    weights = {
                        model_types[0]: w1,
                        model_types[1]: w2,
                        model_types[2]: w3
                    }
                    schemes[name] = weights
    
    # Add standard focus schemes for clarity
    schemes['cnn3d_focus'] = {model: 1.0 if model == 'cnn3d' else 0.2 for model in model_types}
    schemes['vit_focus'] = {model: 1.0 if model == 'vit' else 0.2 for model in model_types}
    schemes['mamba_focus'] = {model: 1.0 if model == 'mamba' else 0.2 for model in model_types}
    
    # Add single model variants for baseline comparison
    for model in model_types:
        name = f'{model}_only'
        weights = {m: 1.0 if m == model else 0.0 for m in model_types}
        schemes[name] = weights
    
    print(f"Generated {len(schemes)} different weighting schemes")
    return schemes

def evaluate_ensemble_with_schemes(data_dir, models_dir, output_dir, device, seq_length=None):
    """
    Evaluate ensemble models with different weighting schemes
    
    Args:
        data_dir: Directory containing the data
        models_dir: Directory containing trained models
        output_dir: Directory to save results
        device: Device to use for evaluation
        seq_length: Sequence length to use for evaluation (default: maximum available)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Model types to include
    model_types = ['cnn3d', 'vit', 'mamba']
    
    # Generate different weighting schemes
    weight_schemes = generate_weight_schemes(model_types)
    
    # Also add metric-based weighting
    results_path = os.path.join(models_dir, "final_results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        
        # Add MIoU-based weights
        miou_values = results_df.set_index('model')['val_miou'].to_dict()
        weight_schemes['miou_weighted'] = miou_values
        
        # Add accuracy-based weights if available
        if 'val_acc' in results_df.columns:
            acc_values = results_df.set_index('model')['val_acc'].to_dict()
            weight_schemes['accuracy_weighted'] = acc_values
    
    # Load model configurations
    model_configs = {}
    models = {}
    
    for model_type in model_types:
        config_path = os.path.join(models_dir, model_type, f"{model_type}_config.csv")
        config_df = pd.read_csv(config_path)
        hyperparams = config_df.iloc[0].to_dict()
        model_configs[model_type] = hyperparams
        
        # Load model
        model_path = os.path.join(models_dir, model_type, f"{model_type}_final.pth")
        models[model_type] = load_model(model_type, hyperparams, model_path, device)
    
    # Determine common parameters for the ensemble
    common_params = determine_common_parameters(model_configs)
    
    # Get parameters for evaluation
    window_size = common_params['window_size']
    window_step = common_params['window_step']
    max_seq_length = common_params['max_seq_length']
    
    # Use provided sequence length or maximum available
    if seq_length is None:
        seq_length = max_seq_length
    else:
        seq_length = min(seq_length, max_seq_length)
    
    print(f"Common parameters: window_size={window_size}, window_step={window_step}")
    print(f"Using sequence length: {seq_length} (max available: {max_seq_length})")
    
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
    
    # Prepare results storage
    all_results = []
    
    # Process each weighting scheme
    for scheme_name, weights in weight_schemes.items():
        print(f"\n{'='*80}")
        print(f"Evaluating ensemble with {scheme_name} weighting scheme")
        
        # Display weights
        for model_name, weight in weights.items():
            print(f"  {model_name}: {weight:.4f}")
        
        print(f"{'='*80}")
        
        # Create ensemble with this weighting scheme
        ensemble = EnsembleModel(models, weights)
        
        # Evaluate ensemble
        metrics = evaluate_metrics(ensemble, val_loader, device)
        
        # Record results
        result = {
            'weighting_scheme': scheme_name,
            'sequence_length': seq_length,
            **metrics
        }
        all_results.append(result)
        
        print(f"Results for {scheme_name}: "
              f"MIoU={metrics['miou']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "ensemble_weighting_schemes.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create visualizations
    plot_metrics_by_scheme(results_df, seq_length, output_dir)
    
    # Also evaluate individual models for comparison
    evaluate_individual_models(models, val_loader, device, output_dir, seq_length)
    
    return results_df

def evaluate_individual_models(models, val_loader, device, output_dir, seq_length):
    """
    Evaluate individual models for comparison with ensemble
    
    Args:
        models: Dictionary of individual models
        val_loader: DataLoader for validation
        device: Device to use
        output_dir: Output directory
        seq_length: Sequence length used
    """
    print("\n=== Evaluating Individual Models for Comparison ===")
    individual_results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_metrics(model, val_loader, device)
        
        result = {
            'model': model_name,
            'sequence_length': seq_length,
            **metrics
        }
        individual_results.append(result)
        
        print(f"Results for {model_name}: "
              f"MIoU={metrics['miou']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}")
    
    # Save results
    individual_df = pd.DataFrame(individual_results)
    individual_path = os.path.join(output_dir, "individual_models_metrics.csv")
    individual_df.to_csv(individual_path, index=False)
    
    # Compare with best ensemble
    compare_with_best_ensemble(individual_df, output_dir)

def compare_with_best_ensemble(individual_df, output_dir):
    """
    Compare individual models with the best ensemble
    
    Args:
        individual_df: DataFrame with individual model results
        output_dir: Output directory
    """
    # Load ensemble results
    ensemble_path = os.path.join(output_dir, "ensemble_weighting_schemes.csv")
    if not os.path.exists(ensemble_path):
        print("Ensemble results not found. Skipping comparison.")
        return
    
    ensemble_df = pd.read_csv(ensemble_path)
    
    # Find best ensemble for each metric
    metrics = ['miou', 'f1', 'precision', 'recall', 'accuracy']
    best_ensembles = {}
    
    for metric in metrics:
        best_idx = ensemble_df[metric].idxmax()
        best_scheme = ensemble_df.loc[best_idx, 'weighting_scheme']
        best_value = ensemble_df.loc[best_idx, metric]
        best_ensembles[metric] = (best_scheme, best_value)
    
    # Create comparison plots
    create_comparison_plots(individual_df, ensemble_df, best_ensembles, output_dir)

def create_comparison_plots(individual_df, ensemble_df, best_ensembles, output_dir):
    """
    Create plots comparing individual models with best ensembles
    
    Args:
        individual_df: DataFrame with individual model results
        ensemble_df: DataFrame with ensemble results
        best_ensembles: Dictionary mapping metrics to (best_scheme, best_value)
        output_dir: Output directory
    """
    metrics = ['miou', 'f1', 'precision', 'recall', 'accuracy']
    
    # Create a bar chart comparing individual models with best ensemble for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        # Get individual model values
        models = individual_df['model'].values
        values = individual_df[metric].values
        
        # Get best ensemble value
        best_scheme, best_value = best_ensembles[metric]
        
        # Create data for plot
        x_labels = list(models) + [f'Best Ensemble\n({best_scheme})']
        y_values = list(values) + [best_value]
        
        # Set colors - regular for individual models, highlight for best ensemble
        colors = ['#1f77b4'] * len(models) + ['#ff7f0e']
        
        # Create bar chart
        bars = axes[i].bar(x_labels, y_values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=45, fontsize=8
            )
        
        # Set title and labels
        axes[i].set_title(metric.upper())
        axes[i].set_ylim(0, 1.0)
        axes[i].set_xticklabels(x_labels, rotation=45, ha='right')
        
    plt.tight_layout()
    plt.suptitle('Individual Models vs. Best Ensemble', y=1.05)
    
    # Save plot
    plot_path = os.path.join(output_dir, "models_vs_best_ensemble.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def plot_metrics_by_scheme(results_df, seq_length, output_dir):
    """
    Create bar plots comparing metrics across different weighting schemes
    for a specific sequence length
    
    Args:
        results_df: DataFrame with results
        seq_length: Sequence length to plot
        output_dir: Directory to save plots
    """
    # Filter for the specific sequence length
    seq_df = results_df[results_df['sequence_length'] == seq_length]
    
    # Get metrics and schemes
    metrics = ['miou', 'f1', 'precision', 'recall', 'accuracy']
    schemes = seq_df['weighting_scheme'].values
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Sort by metric value for better visualization
        sorted_df = seq_df.sort_values(by=metric, ascending=False)
        sorted_schemes = sorted_df['weighting_scheme'].values
        sorted_values = sorted_df[metric].values
        
        # Create bar chart
        bars = axes[i].bar(sorted_schemes, sorted_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=45, fontsize=8
            )
        
        # Set title and labels
        axes[i].set_title(metric.upper())
        axes[i].set_ylim(0, 1.0)
        axes[i].set_xticklabels(sorted_schemes, rotation=45, ha='right')
        
    plt.tight_layout()
    plt.suptitle(f'Metrics by Weighting Scheme (Sequence Length = {seq_length})', y=1.05)
    
    # Save plot
    plot_path = os.path.join(output_dir, f"metrics_by_scheme_seq{seq_length}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # Create radar chart for comprehensive comparison
    create_radar_chart(seq_df, output_dir, seq_length)

def create_radar_chart(seq_df, output_dir, seq_length):
    """
    Create a radar chart comparing all metrics across weighting schemes
    
    Args:
        seq_df: DataFrame with results for a specific sequence length
        output_dir: Directory to save plot
        seq_length: Sequence length used
    """
    # Metrics to include in radar chart
    metrics = ['miou', 'f1', 'precision', 'recall', 'accuracy']
    schemes = seq_df['weighting_scheme'].values
    
    # Number of metrics (variables)
    N = len(metrics)
    
    # Create angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metrics to chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw metric labels at the right position
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], size=10)
    plt.ylim(0, 1)
    
    # Plot each weighting scheme
    for scheme in schemes:
        scheme_data = seq_df[seq_df['weighting_scheme'] == scheme]
        values = [scheme_data[metric].values[0] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, label=scheme)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Comparative Performance of Weighting Schemes (Sequence Length = {seq_length})')
    
    # Save radar chart
    radar_path = os.path.join(output_dir, f"radar_comparison_seq{seq_length}.png")
    plt.savefig(radar_path, bbox_inches='tight')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble model with different weighting schemes')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='results/final_models', help='Path to models directory')
    parser.add_argument('--output_dir', type=str, default='results/ensemble', help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seq_length', type=int, default=None, help='Sequence length to use for evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate ensemble with different weighting schemes
    print("\n=== Comparing different ensemble weighting schemes ===")
    results_df = evaluate_ensemble_with_schemes(
        args.data_dir,
        args.models_dir,
        args.output_dir,
        device,
        args.seq_length
    )
    
    print("\nWeighting scheme comparison complete. Results saved to", args.output_dir)

def evaluate_metrics(model, data_loader, device, threshold=0.5):
    """
    Evaluate multiple metrics on a dataset: MIoU, Precision, Recall, F1 Score
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with validation data
        device: Device to use
        threshold: Probability threshold for positive class (defect)
    """
    model.eval()
    
    # Initialize accumulators for pixel-wise segmentation metrics
    tp_total = 0  # true positive
    fp_total = 0  # false positive
    fn_total = 0  # false negative
    tn_total = 0  # true negative
    
    # Initialize IoU accumulators 
    num_classes = 2
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    # Debug variables
    total_pixels = 0
    total_positive_preds = 0
    total_positive_targets = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Calculating Metrics'):
            # Process batch
            data = data.to(device)
            target = target.to(device)
            
            # Get predictions (probabilities)
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            
            # Print probability statistics for debugging
            if total_pixels == 0:  # Only for first batch
                p_min = probabilities[:, 1].min().item()
                p_max = probabilities[:, 1].max().item()
                p_mean = probabilities[:, 1].mean().item()
                print(f"Debug - Probabilities: min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.4f}")
            
            # Get probability of defect class (class 1)
            defect_probs = probabilities[:, 1]  # Shape: [batch_size, height, width]
            
            # Apply threshold to get binary predictions
            pred_positive = (defect_probs > threshold)  # Predicted defect pixels
            
            # Convert target to binary mask (ensure it's actually binary)
            target_binary = target.clone()
            target_binary[target > 0] = 1  # Convert any positive value to 1
            target_positive = (target_binary == 1)  # Actual defect pixels
            
            # Count pixels for debugging
            batch_pixels = target.numel()
            total_pixels += batch_pixels
            total_positive_preds += pred_positive.sum().item()
            total_positive_targets += target_positive.sum().item()
            
            # Calculate confusion matrix elements (pixel-wise)
            tp = (pred_positive & target_positive).sum().item()  # True positives
            fp = (pred_positive & ~target_positive).sum().item()  # False positives
            fn = (~pred_positive & target_positive).sum().item()  # False negatives
            tn = (~pred_positive & ~target_positive).sum().item()  # True negatives
            
            # Accumulate totals
            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
            
            # Calculate IoU for each class using the same threshold approach
            # Background class (0)
            pred_background = ~pred_positive
            target_background = ~target_positive
            intersection[0] += torch.logical_and(pred_background, target_background).sum().item()
            union[0] += torch.logical_or(pred_background, target_background).sum().item()
            
            # Defect class (1)
            intersection[1] += torch.logical_and(pred_positive, target_positive).sum().item()
            union[1] += torch.logical_or(pred_positive, target_positive).sum().item()
    
    # Print detailed debug info
    print(f"\nDetailed metrics information:")
    print(f"Total pixels evaluated: {total_pixels}")
    print(f"Positive predictions: {total_positive_preds} ({total_positive_preds/total_pixels:.2%} of total)")
    print(f"Positive targets: {total_positive_targets} ({total_positive_targets/total_pixels:.2%} of total)")
    print(f"TP: {tp_total}, FP: {fp_total}, FN: {fn_total}, TN: {tn_total}")
    
    # Calculate metrics from accumulated totals
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (tp_total + tn_total + fp_total + fn_total) > 0 else 0.0
    
    # Calculate IoU for each class
    iou = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        if union[cls] > 0:
            iou[cls] = intersection[cls] / union[cls]
        else:
            iou[cls] = 1.0
    
    # Mean IoU across classes
    miou = iou.mean().item()
    
    # Return all metrics
    return {
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'pos_iou': iou[1].item()  # IoU for the positive class (defects)
    }

if __name__ == "__main__":
    main() 
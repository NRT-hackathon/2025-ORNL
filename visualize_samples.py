import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import get_data_loaders, CrossValidationMode
from create_test_data import create_test_data

def display_sample_info(loader, num_samples=10, title="Samples"):
    """Display information about samples in the loader."""
    samples = []
    
    print(f"\n{title}:")
    for i, (data, target) in enumerate(loader):
        if i >= num_samples:
            break
            
        # Print shapes
        print(f"Sample {i+1}:")
        print(f"  Input shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Sequence length: {data.shape[1]}")
        print(f"  Input min/max: {data.min().item():.2f}/{data.max().item():.2f}")
        print(f"  Target unique values: {torch.unique(target).tolist()}")
        
        # Store sample for visualization
        samples.append((data, target))
    
    return samples

def visualize_sample(samples, sample_idx=0):
    """Visualize a single sample, handling different sequence lengths."""
    data, target = samples[sample_idx]
    seq_len = data.shape[1]
    
    # Handle different sequence lengths
    if seq_len == 1:
        # For sequence length 1, use a simpler layout
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Plot the single frame
        axes[0].imshow(data[0, 0].numpy(), cmap='gray')
        axes[0].set_title("Frame 1")
        axes[0].axis('off')
        
        # Plot the target
        im = axes[1].imshow(target[0].numpy(), cmap='viridis')
        axes[1].set_title("Target Label")
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1])
    else:
        # For sequence length > 1, use the original layout
        fig, axes = plt.subplots(2, seq_len, figsize=(4*seq_len, 8))
        
        # Plot each frame in the sequence
        for i in range(seq_len):
            axes[0, i].imshow(data[0, i].numpy(), cmap='gray')
            axes[0, i].set_title(f"Frame {i+1}")
            axes[0, i].axis('off')
        
        # Plot the target
        center_idx = seq_len//2
        im = axes[1, center_idx].imshow(target[0].numpy(), cmap='viridis')
        axes[1, center_idx].set_title("Target Label")
        axes[1, center_idx].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, center_idx])
        
        # Hide unused axes
        for i in range(seq_len):
            if i != center_idx:
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"sample_{sample_idx}_seq{seq_len}.png")
    plt.close()

def visualize_seq_lengths(loader, max_samples=100):
    """Visualize samples with different sequence lengths."""
    sequence_lengths_found = set()
    
    for i, (data, target) in enumerate(loader):
        seq_len = data.shape[1]
        
        # Only visualize each unique sequence length once
        if seq_len not in sequence_lengths_found and len(sequence_lengths_found) < 5:
            sequence_lengths_found.add(seq_len)
            print(f"\nSample with sequence length {seq_len}:")
            
            # For sequence length 1
            if seq_len == 1:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                
                # Plot the single frame
                axes[0].imshow(data[0, 0].numpy(), cmap='gray')
                axes[0].set_title("Frame 1")
                axes[0].axis('off')
                
                # Plot the target
                im = axes[1].imshow(target[0].numpy(), cmap='viridis')
                axes[1].set_title("Target Label")
                axes[1].axis('off')
                
                plt.colorbar(im, ax=axes[1])
            else:
                # For sequence length > 1
                fig, axes = plt.subplots(1, seq_len + 1, figsize=(3*(seq_len + 1), 3))
                
                # Plot each frame in the sequence
                for j in range(seq_len):
                    axes[j].imshow(data[0, j].numpy(), cmap='gray')
                    axes[j].set_title(f"Frame {j+1}")
                    axes[j].axis('off')
                
                # Plot the target
                im = axes[-1].imshow(target[0].numpy(), cmap='viridis')
                axes[-1].set_title("Target")
                axes[-1].axis('off')
                
                plt.colorbar(im, ax=axes[-1])
            
            plt.tight_layout()
            plt.savefig(f"sequence_length_{seq_len}.png")
            plt.close()
        
        # Stop after checking a reasonable number of samples
        if i >= max_samples:
            break

if __name__ == "__main__":
    # Create test data if it doesn't exist
    if not os.path.exists('data/frames'):
        create_test_data(num_frames=15, image_size=(256, 256))
    
    # Initialize data loaders
    data_dir = "data"
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        window_size=64,     # Smaller windows for testing
        window_step=32,     # Larger step for fewer windows 
        sequence_length=5,  # Stack up to 5 frames
        cross_val_mode=CrossValidationMode.TOP,
        batch_size=1,       # Set to 1 to see individual samples
        num_workers=0       # Set to 0 for easier debugging
    )
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Get sample information
    train_samples = display_sample_info(train_loader, num_samples=10, title="Training Samples")
    
    # Visualize first 3 samples
    for i in range(min(3, len(train_samples))):
        visualize_sample(train_samples, i)
    
    # Visualize different sequence lengths
    visualize_seq_lengths(train_loader)
    
    print(f"\nVisualization complete. Check the current directory for saved sample images.") 
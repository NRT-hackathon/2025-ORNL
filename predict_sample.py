import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import TemporalWindowDataset, CrossValidationMode
from model.cnn3d import CNN3D
from model.mamba_cnn import MambaCNN
from model.vit import ViTModel

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions on random samples')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='output', help='Path to trained models directory')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Path to save prediction results')
    parser.add_argument('--window_size', type=int, default=64, help='Window size for patches')
    parser.add_argument('--window_step', type=int, default=8, help='Step size for sliding window')
    parser.add_argument('--max_seq_length', type=int, default=5, help='Maximum sequence length')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to predict')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for prediction')
    return parser.parse_args()

def load_model(model_type, model_path, device):
    """Load a trained model"""
    if model_type == 'cnn3d':
        model = CNN3D(max_seq_length=5, in_channels=1, out_channels=2)
    elif model_type == 'mamba':
        model = MambaCNN(in_channels=1, out_channels=2)
    elif model_type == 'vit':
        model = ViTModel(in_channels=1, out_channels=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_defects(prediction, threshold=0.5, bbox=16):
    """Extract defects from prediction map using connected components"""
    import cv2
    from scipy import ndimage
    
    # Threshold predictions
    _, thresh = cv2.threshold(prediction, threshold, 1, cv2.THRESH_BINARY)
    
    # Find connected components
    s = [[1,1,1],[1,1,1],[1,1,1]]
    labeled, nr_objects = ndimage.label(thresh, structure=s)
    cc = ndimage.measurements.center_of_mass(thresh, labeled, range(1, nr_objects + 1))
    sizes = ndimage.sum(thresh, labeled, range(1, nr_objects + 1))
    
    # Filter found points by size
    cc = np.array(cc)
    sizes = np.array(sizes)
    valid_indices = (sizes > 5) & (sizes < 700)
    centers = cc[valid_indices]
    
    return centers

def visualize_predictions(image, predictions, centers, output_path, model_name):
    """Visualize predictions with bounding boxes"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot prediction with bounding boxes
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(predictions, alpha=0.3, cmap='hot')
    axes[1].set_title(f'{model_name} Prediction')
    axes[1].axis('off')
    
    # Add bounding boxes
    bbox = 16
    for center in centers:
        startx = int(round(center[0] - bbox))
        starty = int(round(center[1] - bbox))
        rect = patches.Rectangle((starty, startx), bbox*2, bbox*2, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = TemporalWindowDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        window_step=args.window_step,
        max_seq_length=args.max_seq_length,
        cross_val_mode=CrossValidationMode.NONE
    )
    
    # Load models
    models = {}
    for model_type in ['cnn3d', 'mamba', 'vit']:
        model_path = os.path.join(args.models_dir, f'{model_type}_best_acc.pth')
        if os.path.exists(model_path):
            models[model_type] = load_model(model_type, model_path, args.device)
    
    if not models:
        raise ValueError("No trained models found in the models directory")
    
    # Get random samples
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    
    for idx in indices:
        # Get sample
        sample = dataset[idx]
        frames = sample['frames']  # [seq_len, height, width]
        label = sample['label']    # [height, width]
        
        # Make predictions with each model
        for model_name, model in models.items():
            with torch.no_grad():
                # Prepare input
                x = torch.from_numpy(frames).unsqueeze(0).to(args.device)  # [1, seq_len, height, width]
                
                # Get prediction
                pred = model(x)
                pred = torch.softmax(pred, dim=1)[0, 1].cpu().numpy()  # Get probability of defect class
                
                # Extract defects
                centers = extract_defects(pred)
                
                # Visualize
                output_path = os.path.join(args.output_dir, f'sample_{idx}_{model_name}.png')
                visualize_predictions(frames[-1], pred, centers, output_path, model_name)

if __name__ == '__main__':
    main() 
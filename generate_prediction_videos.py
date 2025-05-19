import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from dataset import TemporalWindowDataset, CrossValidationMode
from model.cnn3d import CNN3D
from model.vit import VisionTransformer
from model.mamba_cnn import OptimizedMambaCNN as MambaCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Generate prediction videos for all models')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='results/final_models', help='Path to trained models directory')
    parser.add_argument('--output_dir', type=str, default='prediction_videos', help='Path to save prediction videos')
    parser.add_argument('--ablation_file', type=str, default='results/ablation/ablation.csv', 
                       help='Path to ablation study results file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for prediction')
    parser.add_argument('--overlay_alpha', type=float, default=0.3, 
                       help='Alpha for overlay of predictions on original frames')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for output video')
    parser.add_argument('--ensemble', action='store_true', help='Generate ensemble prediction video')
    parser.add_argument('--full_heatmap', action='store_true', 
                       help='Show full heatmap visualization instead of thresholded defects')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug visualizations for troubleshooting')
    return parser.parse_args()

def load_ablation_results(ablation_file):
    """Load best hyperparameters from ablation study results"""
    with open(ablation_file, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    # Convert string values to appropriate types
    for result in results:
        result['window_size'] = int(result['window_size'])
        result['window_step'] = int(result['window_step'])
        result['max_seq_length'] = int(result['max_seq_length'])
        result['avg_val_miou'] = float(result['avg_val_miou'])
    
    # Get the best hyperparameters for each model type
    best_params = {}
    for result in results:
        model_type = result['model']
        if model_type not in best_params or result['avg_val_miou'] > best_params[model_type]['avg_val_miou']:
            best_params[model_type] = result
    
    return best_params

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

def get_sliding_windows(frame, window_size, window_step):
    """Generate sliding windows from a frame with positions"""
    height, width = frame.shape
    windows = []
    positions = []
    
    # Use a proper grid of positions to ensure complete coverage
    for y in range(0, height - window_size + 1, window_step):
        for x in range(0, width - window_size + 1, window_step):
            window = frame[y:y+window_size, x:x+window_size]
            windows.append(window)
            positions.append((y, x))
            
    return np.array(windows), positions

def create_temporal_windows(frames, window_size, window_step, max_seq_length):
    """Create temporal windows for all frames"""
    all_temporal_windows = []
    all_positions = []
    
    # Print the shape and dimensions of the frames
    print(f"Frame shape: {frames[0].shape}")
    print(f"Total frames: {len(frames)}")
    print(f"Window parameters: size={window_size}, step={window_step}, seq_length={max_seq_length}")
    
    # Calculate how many windows we should expect
    height, width = frames[0].shape
    n_windows_h = 1 + (height - window_size) // window_step
    n_windows_w = 1 + (width - window_size) // window_step
    n_frames = len(frames) - max_seq_length + 1
    expected_windows = n_windows_h * n_windows_w * n_frames
    
    print(f"Expected windows per frame: {n_windows_h}x{n_windows_w} = {n_windows_h * n_windows_w}")
    print(f"Expected total windows: {expected_windows}")
    
    for i in range(len(frames) - max_seq_length + 1):
        frame_sequence = frames[i:i+max_seq_length]
        
        # Use the last frame in the sequence for windowing
        last_frame = frame_sequence[-1]
        windows, positions = get_sliding_windows(last_frame, window_size, window_step)
        
        # For each window position, get the temporal sequence
        for j, (y, x) in enumerate(positions):
            temporal_window = []
            for frame in frame_sequence:
                window = frame[y:y+window_size, x:x+window_size]
                temporal_window.append(window)
            
            # Stack temporal sequence
            temporal_window = np.stack(temporal_window)
            all_temporal_windows.append(temporal_window)
            all_positions.append((i + max_seq_length - 1, y, x))  # (frame_idx, y, x)
    
    print(f"Actually created windows: {len(all_temporal_windows)}")
    
    # Verify window positions for the first frame
    first_frame_idx = min([pos[0] for pos in all_positions])
    first_frame_positions = [(y, x) for (frame_idx, y, x) in all_positions if frame_idx == first_frame_idx]
    print(f"Windows for first frame: {len(first_frame_positions)}")
    print(f"Window positions cover from ({min([y for y, _ in first_frame_positions])}, {min([x for _, x in first_frame_positions])}) "
          f"to ({max([y for y, _ in first_frame_positions]) + window_size}, {max([x for _, x in first_frame_positions]) + window_size})")
    
    return all_temporal_windows, all_positions

def predict_with_model(model, temporal_windows, device):
    """Make predictions with proper output handling"""
    predictions = []
    
    batch_size = 32
    n_batches = len(temporal_windows) // batch_size + (1 if len(temporal_windows) % batch_size != 0 else 0)
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=f"Predicting with {type(model).__name__}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(temporal_windows))
            
            batch = temporal_windows[start_idx:end_idx]
            batch = np.array(batch)
            
            # Convert batch to proper format [B, T, H, W]
            if len(batch.shape) == 5:  # [B, T, C, H, W]
                batch = batch.squeeze(2)  # Remove channel dimension if it exists
            
            # Create tensor and move to device
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            # Print shape for debugging
            print(f"Input tensor shape: {batch_tensor.shape}")
            
            outputs = model(batch_tensor)
            
            # Check output dimensionality
            print(f"Raw model output shape: {outputs.shape}")
            
            # Handle different output formats
            if len(outputs.shape) == 4:  # [B, C, H, W]
                probs = F.softmax(outputs, dim=1)
                pred_maps = probs[:, 1].cpu().numpy()  # Get class 1 predictions
            elif len(outputs.shape) == 2:  # [B, C]
                probs = F.softmax(outputs, dim=1)
                pred_maps = probs[:, 1].cpu().numpy()  # Single value per window
                # Need to expand to window size
                window_size = batch.shape[-1]
                expanded_pred_maps = []
                for pred in pred_maps:
                    pred_map = np.full((window_size, window_size), pred)
                    expanded_pred_maps.append(pred_map)
                pred_maps = np.array(expanded_pred_maps)
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            
            # Verify prediction shape
            print(f"Processed prediction shape: {pred_maps.shape}")
            
            for pred_map in pred_maps:
                predictions.append(pred_map)
    
    return predictions

def assemble_prediction_map(predictions, positions, frame_shape, window_size, window_step):
    """Fixed assembly with proper position tracking"""
    frame_indices = sorted(set([pos[0] for pos in positions]))
    frame_maps = {}
    
    for frame_idx in frame_indices:
        # Create accumulators for this frame
        sum_pred = np.zeros(frame_shape, dtype=np.float32)
        count_pred = np.zeros(frame_shape, dtype=np.float32)
        
        # Get all predictions for this frame
        frame_positions = [(i, y, x) for i, (fidx, y, x) in enumerate(positions) if fidx == frame_idx]
        
        print(f"Frame {frame_idx}: Processing {len(frame_positions)} windows")
        
        for window_idx, y, x in frame_positions:
            window_pred = predictions[window_idx]
            
            # Verify prediction shape matches expected window size
            assert window_pred.shape == (window_size, window_size), \
                f"Prediction shape {window_pred.shape} doesn't match window size {window_size}"
            
            # Calculate the region of the frame to update
            y_end = min(y + window_size, frame_shape[0])
            x_end = min(x + window_size, frame_shape[1])
            
            # Calculate the region of the window to use
            window_h = y_end - y
            window_w = x_end - x
            
            # Place the prediction in the frame, handling edge cases
            sum_pred[y:y_end, x:x_end] += window_pred[:window_h, :window_w]
            count_pred[y:y_end, x:x_end] += 1
        
        # Average overlapping predictions
        mask = count_pred > 0
        prediction_map = np.zeros(frame_shape, dtype=np.float32)
        prediction_map[mask] = sum_pred[mask] / count_pred[mask]
        
        # Check coverage
        coverage = (count_pred > 0).sum() / (frame_shape[0] * frame_shape[1])
        print(f"Frame {frame_idx}: Coverage = {coverage:.2%}")
        
        frame_maps[frame_idx] = prediction_map
    
    return frame_maps

def visualize_window_coverage(frame_shape, positions, window_size, window_step, frame_idx, output_path):
    """Visualize the coverage of windows for debugging"""
    # Create blank image
    coverage = np.zeros(frame_shape, dtype=np.float32)
    count = np.zeros(frame_shape, dtype=np.float32)
    
    # Add each window position
    for pos_frame_idx, y, x in positions:
        if pos_frame_idx == frame_idx:
            coverage[y:y+window_size, x:x+window_size] += 1
            # Mark window centers more brightly
            center_y, center_x = y + window_size//2, x + window_size//2
            coverage[center_y-1:center_y+1, center_x-1:center_x+1] += 3
    
    # Normalize and create visualization
    if np.max(coverage) > 0:
        coverage = coverage / np.max(coverage)
    
    # Create color image
    colored = cv2.applyColorMap(np.uint8(coverage * 255), cv2.COLORMAP_JET)
    
    # Add grid lines to show window steps
    for y in range(0, frame_shape[0], window_step):
        colored[y, :] = [255, 255, 255]
    for x in range(0, frame_shape[1], window_step):
        colored[:, x] = [255, 255, 255]
    
    # Save the visualization
    cv2.imwrite(output_path, colored)
    print(f"Saved window coverage visualization to {output_path}")

def visualize_assembly_debug(frame, predictions, positions, window_size, window_step, frame_idx, output_path):
    """Visualize the assembly process for debugging"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0,0].imshow(frame, cmap='gray')
    axes[0,0].set_title('Original Frame')
    
    # Window positions
    axes[0,1].imshow(frame, cmap='gray')
    frame_positions = [(y, x) for (fidx, y, x) in positions if fidx == frame_idx]
    for i, (y, x) in enumerate(frame_positions):
        rect = plt.Rectangle((x, y), window_size, window_size, 
                            fill=False, edgecolor='red', linewidth=2)
        axes[0,1].add_patch(rect)
        axes[0,1].text(x, y, str(i), color='yellow')
    axes[0,1].set_title('Window Positions')
    
    # Prediction values for each window
    pred_values = []
    for i, (y, x) in enumerate(frame_positions):
        window_idx = [idx for idx, (fidx, py, px) in enumerate(positions) 
                     if fidx == frame_idx and py == y and px == x][0]
        pred_values.append(predictions[window_idx].mean())
    axes[0,2].bar(range(len(pred_values)), pred_values)
    axes[0,2].set_title('Average Prediction per Window')
    
    # Assembled prediction
    # Create a mapping of position indices to prediction indices
    window_indices = {}
    for i, (fidx, y, x) in enumerate(positions):
        if fidx == frame_idx:
            window_indices[(y, x)] = i
    
    assembled = assemble_prediction_map([predictions[window_indices.get((y, x), 0)] for y, x in frame_positions],
                                         [(frame_idx, y, x) for y, x in frame_positions],
                                         frame.shape, window_size, window_step)
    
    axes[1,0].imshow(assembled[frame_idx], cmap='hot')
    axes[1,0].set_title('Assembled Prediction')
    
    # Coverage map
    coverage = np.zeros(frame.shape)
    for y, x in frame_positions:
        y_end = min(y + window_size, frame.shape[0])
        x_end = min(x + window_size, frame.shape[1])
        coverage[y:y_end, x:x_end] += 1
    
    axes[1,1].imshow(coverage, cmap='viridis')
    axes[1,1].set_title('Coverage Map')
    
    # Overlay
    overlay = frame.copy()
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(assembled[frame_idx] * 255), cv2.COLORMAP_JET)
    overlay_result = cv2.addWeighted(overlay_rgb, 0.7, heatmap, 0.3, 0)
    
    axes[1,2].imshow(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title('Overlay')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# def assemble_prediction_map(predictions, positions, frame_shape, window_size, window_step, output_dir=None):
#     """
#     Assemble prediction map from per-pixel window predictions using weighted voting.
#     """
#     frame_indices = sorted(set([pos[0] for pos in positions]))
#     frame_maps = {}
    
#     for frame_idx in frame_indices:
#         # Initialize accumulation arrays for weighted voting
#         weighted_sum = np.zeros(frame_shape, dtype=np.float32)
#         weight_sum = np.zeros(frame_shape, dtype=np.float32)
        
#         frame_predictions = [(i, y, x) for i, (fidx, y, x) in enumerate(positions) if fidx == frame_idx]
        
#         for window_idx, y, x in frame_predictions:
#             # Get the per-pixel prediction for this window
#             window_pred = predictions[window_idx]  # Shape: [H, W]
            
#             # Calculate confidence weights (optional)
#             confidence = np.abs(window_pred - 0.5) * 2  # Normalize to [0, 1]
            
#             # Add window prediction to the correct region
#             y_end = min(y + window_size, frame_shape[0])
#             x_end = min(x + window_size, frame_shape[1])
            
#             # Handle edge cases where window might extend beyond frame
#             window_h = y_end - y
#             window_w = x_end - x
            
#             weighted_sum[y:y_end, x:x_end] += window_pred[:window_h, :window_w] * confidence[:window_h, :window_w]
#             weight_sum[y:y_end, x:x_end] += confidence[:window_h, :window_w]
        
#         # Average the weighted accumulation
#         mask = weight_sum > 0
#         prediction_map = np.zeros(frame_shape, dtype=np.float32)
#         prediction_map[mask] = weighted_sum[mask] / weight_sum[mask]
        
#         # Handle areas with no predictions
#         low_confidence_mask = weight_sum < 0.1
#         prediction_map[low_confidence_mask] = 0
        
#         frame_maps[frame_idx] = prediction_map
    
#     return frame_maps

def overlay_prediction(frame, prediction, alpha=0.3, threshold=0.5, full_heatmap=False):
    """Overlay prediction on the original frame with a color gradient based on prediction probability"""
    # Convert grayscale frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Create a heatmap from the prediction
    heatmap = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    if full_heatmap:
        # Apply a full gradient heatmap visualization (blue -> green -> yellow -> red)
        # Create a colormap for visualization
        # 0 (no defect) -> blue, 0.3 -> green, 0.6 -> yellow, 1.0 (defect) -> red
        
        # Normalize the prediction probabilities to 0-255 range for color mapping
        normalized = np.uint8(prediction * 255)
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Only apply colormap where the prediction is somewhat significant
        # This avoids coloring the entire image with low-value noise
        mask = prediction > 0.05
        heatmap[mask] = colored_heatmap[mask]
    else:
        # Get mask of pixels above threshold
        pred_mask = prediction > threshold
        
        if np.any(pred_mask):
            # Normalize prediction values in the defect regions to 0-1 range for color mapping
            # Values closer to threshold will be mapped to 0, values closer to 1 will be mapped to 1
            normalized_pred = np.zeros_like(prediction)
            normalized_pred[pred_mask] = (prediction[pred_mask] - threshold) / (1.0 - threshold)
            
            # Apply color gradient: yellow (low confidence) to red (high confidence)
            # Red channel - always high for defect areas
            heatmap[pred_mask, 2] = 255  # R
            
            # Green channel - high for low confidence (yellow), low for high confidence (red)
            # Map normalized predictions inversely to green intensity
            heatmap[pred_mask, 1] = np.uint8(255 * (1.0 - normalized_pred[pred_mask]))  # G
            
            # Blue channel - always zero for this color scheme
            heatmap[pred_mask, 0] = 0  # B
    
    # Apply alpha blending
    overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap, alpha, 0)
    
    return overlay

def create_prediction_video(frames, predictions, output_path, alpha=0.3, threshold=0.5, fps=5, full_heatmap=False):
    """Create a video of the predictions overlaid on the original frames"""
    if not frames or not predictions:
        print(f"No frames or predictions available for {output_path}")
        return
    
    # Get the first available frame index
    first_frame_idx = min(predictions.keys())
    
    # Get frame dimensions
    height, width = frames[0].shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for i, frame in enumerate(frames):
        if i in predictions:
            # Create overlay with prediction
            overlay = overlay_prediction(frame, predictions[i], alpha, threshold, full_heatmap)
            video_writer.write(overlay)
        else:
            # For frames without prediction, just use the original frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            video_writer.write(frame_rgb)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = os.path.join(args.output_dir, "debug") if args.debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Load best hyperparameters from ablation study
    best_params = load_ablation_results(args.ablation_file)
    print("Best hyperparameters from ablation study:")
    for model_type, params in best_params.items():
        print(f"  {model_type}: window_size={params['window_size']}, "
              f"window_step={params['window_step']}, max_seq_length={params['max_seq_length']}")
    
    # Load dataset frames
    print("\nLoading dataset...")
    frames_dir = os.path.join(args.data_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                          if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.npy')])
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        if frame_file.endswith('.npy'):
            frame = np.load(frame_path)
        else:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        frames.append(frame)
    
    print(f"Loaded {len(frames)} frames")
    
    # Dictionary to store all model predictions
    all_predictions = {}
    
    # Process each model
    for model_type, params in best_params.items():
        print(f"\nProcessing {model_type} model...")
        
        # Set hyperparameters
        window_size = params['window_size']
        window_step = params['window_step']
        max_seq_length = params['max_seq_length']
        
        # Create model-specific debug directory if debugging
        model_debug_dir = os.path.join(debug_dir, model_type) if debug_dir else None
        if model_debug_dir:
            os.makedirs(model_debug_dir, exist_ok=True)
        
        # Load the model
        model_path = os.path.join(args.models_dir, f"{model_type}/{model_type}_best_acc.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.models_dir, f"{model_type}_best_acc.pth")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
        
        model = load_model(model_type, params, model_path, args.device)
        
        # Create temporal windows
        print(f"Creating temporal windows with size={window_size}, step={window_step}, seq_length={max_seq_length}...")
        temporal_windows, positions = create_temporal_windows(frames, window_size, window_step, max_seq_length)
        print(f"Created {len(temporal_windows)} temporal windows")
        
        # Make predictions
        predictions = predict_with_model(model, temporal_windows, args.device)
        
        # Debug: check predictions list
        print(f"Total predictions: {len(predictions)}")
        print(f"First prediction shape: {predictions[0].shape}")
        print(f"First prediction stats: mean={predictions[0].mean():.3f}, std={predictions[0].std():.3f}")
        
        # Assemble with debugging
        frame_shape = frames[0].shape
        prediction_maps = assemble_prediction_map(predictions, positions, frame_shape, 
                                                        window_size, window_step)
        # # Make predictions
        # predictions = predict_with_model(model, temporal_windows, args.device)
        
        # # Assemble prediction maps
        # frame_shape = frames[0].shape
        # prediction_maps = assemble_prediction_map(predictions, positions, frame_shape, 
        #                                         window_size, window_step, model_debug_dir)
        
        # Create video
        output_path = os.path.join(args.output_dir, f"{model_type}_predictions.mp4")
        create_prediction_video(frames, prediction_maps, output_path, 
                              alpha=args.overlay_alpha, threshold=args.threshold, 
                              fps=args.fps, full_heatmap=args.full_heatmap)
        
        # Store predictions for ensemble
        all_predictions[model_type] = prediction_maps

        # Visualize assembly process
        debug_frame_idx = max_seq_length - 1
        debug_output = os.path.join(debug_dir, f"{model_type}_assembly_debug_frame_{debug_frame_idx}.png")
        visualize_assembly_debug(frames[debug_frame_idx], predictions, positions, 
                                 window_size, window_step, debug_frame_idx, debug_output)
        
        # Check for common issues:
        print("\nDiagnostics:")
        print(f"1. Window size consistency: {all(p.shape == (window_size, window_size) for p in predictions)}")
        print(f"2. Position frame index range: {min(p[0] for p in positions)} to {max(p[0] for p in positions)}")
        print(f"3. Total windows: {len(temporal_windows)} == {len(predictions)}: {len(temporal_windows) == len(predictions)}")
        
        # Check for edge cases
        frame_positions = [(i, y, x) for i, (fidx, y, x) in enumerate(positions) if fidx == debug_frame_idx]
        edge_windows = [(i, y, x) for i, y, x in frame_positions 
                        if y + window_size > frames[0].shape[0] or x + window_size > frames[0].shape[1]]
        print(f"4. Edge windows for frame {debug_frame_idx}: {len(edge_windows)}")
        
        # Assemble with fixed function
        frame_shape = frames[0].shape
        prediction_maps = assemble_prediction_map(predictions, positions, frame_shape, 
                                                        window_size, window_step)
    
    # Create ensemble prediction video if requested
    if args.ensemble and len(all_predictions) > 1:
        print("\nCreating ensemble prediction video...")
        
        # Create ensemble debug directory if debugging
        ensemble_debug_dir = os.path.join(debug_dir, "ensemble") if debug_dir else None
        if ensemble_debug_dir:
            os.makedirs(ensemble_debug_dir, exist_ok=True)
        
        # Combine predictions from all models with equal weighting
        ensemble_predictions = {}
        
        # Find common frame indices across all models
        common_frames = set()
        for model_preds in all_predictions.values():
            if not common_frames:
                common_frames = set(model_preds.keys())
            else:
                common_frames = common_frames.intersection(set(model_preds.keys()))
        
        print(f"Ensemble will use {len(common_frames)} frames common to all models")
        
        # Average predictions across models
        for frame_idx in common_frames:
            ensemble_pred = np.zeros_like(frames[0], dtype=np.float32)
            count = 0
            
            for model_type, preds in all_predictions.items():
                if frame_idx in preds:
                    ensemble_pred += preds[frame_idx]
                    count += 1
            
            if count > 0:
                ensemble_pred /= count
                ensemble_predictions[frame_idx] = ensemble_pred
                
                # Save debug visualization for ensemble if requested
                if ensemble_debug_dir:
                    ensemble_vis = cv2.applyColorMap(np.uint8(ensemble_pred * 255), cv2.COLORMAP_JET)
                    vis_path = os.path.join(ensemble_debug_dir, f"frame_{frame_idx}_ensemble.png")
                    cv2.imwrite(vis_path, ensemble_vis)
        
        # Create ensemble video
        output_path = os.path.join(args.output_dir, "ensemble_predictions.mp4")
        create_prediction_video(frames, ensemble_predictions, output_path, 
                              alpha=args.overlay_alpha, threshold=args.threshold, 
                              fps=args.fps, full_heatmap=args.full_heatmap)

if __name__ == "__main__":
    main() 
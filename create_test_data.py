import os
import numpy as np
import cv2
from pathlib import Path

def create_test_data(base_dir="data", num_frames=10, image_size=(512, 512)):
    """
    Create synthetic test data for the temporal window dataset.
    
    Args:
        base_dir: Base directory to create data in
        num_frames: Number of frames to generate
        image_size: Size of generated images (height, width)
    """
    # Create directories
    frames_dir = os.path.join(base_dir, "frames")
    labels_dir = os.path.join(base_dir, "labels")
    
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_frames} synthetic frames in {base_dir}")
    
    # Generate frames and labels
    for i in range(num_frames):
        # Create frame with random noise and some patterns
        frame = np.random.rand(*image_size) * 0.4  # Base noise
        
        # Add some circular patterns that move over time
        center_x = int(image_size[1] * 0.5 + 50 * np.sin(i * 0.5))
        center_y = int(image_size[0] * 0.5 + 50 * np.cos(i * 0.5))
        
        # Draw circles
        for r in range(10, 100, 20):
            cv2.circle(frame, (center_x, center_y), r, 1.0, 2)
        
        # Create label - identifying regions
        label = np.zeros(image_size, dtype=np.uint8)
        
        # Add defect regions (class 1)
        for j in range(5):
            x = int(image_size[1] * 0.2) + j * 60 + i * 5
            y = int(image_size[0] * 0.3) + j * 40
            if x < image_size[1] and y < image_size[0]:
                cv2.rectangle(label, (x, y), (x + 30, y + 30), 1, -1)
        
        # Add another type of defect (class 2)
        for j in range(3):
            x = int(image_size[1] * 0.6) - j * 30 - i * 3
            y = int(image_size[0] * 0.7) - j * 20
            if x > 0 and y > 0:
                cv2.circle(label, (x, y), 15, 2, -1)
        
        # Save as numpy arrays
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.npy")
        label_path = os.path.join(labels_dir, f"frame_{i:03d}.npy")
        
        np.save(frame_path, frame)
        np.save(label_path, label)
        
        # Also save as images for visualization
        frame_img = (frame * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame_img)
        
        # Scale label for visualization (multiply by 100 to make it visible)
        label_img = (label * 100).astype(np.uint8)
        cv2.imwrite(os.path.join(labels_dir, f"frame_{i:03d}.png"), label_img)
    
    print(f"Created {num_frames} frames and labels as both .npy and .png files")

if __name__ == "__main__":
    create_test_data(num_frames=15, image_size=(256, 256)) 
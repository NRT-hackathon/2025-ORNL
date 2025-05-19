import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict
import random

class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    NONE = "none"

class TemporalWindowDataset(Dataset):
    """
    Dataset for creating temporal windows of frames.
    
    This dataset uses the same windowing technique from the archive code,
    but stacks windows temporally and supports cross-validation and augmentation.
    """
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 256,
        window_step: int = 8,
        sequence_length: int = 5,
        cross_val_mode: Union[CrossValidationMode, str] = CrossValidationMode.NONE,
        train: bool = True,
        augment: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the processed frames and labels
            window_size: Size of the window to extract
            window_step: Step size for sliding window
            sequence_length: Maximum number of frames to stack temporally
            cross_val_mode: Mode for cross-validation (TOP, LEFT, RIGHT, BOTTOM, NONE)
            train: Whether this is the training set or validation set
            augment: Whether to apply data augmentation
            normalize: Whether to normalize the data
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_step = window_step
        self.sequence_length = sequence_length
        self.train = train
        self.augment = augment
        self.normalize = normalize
        
        # Convert string to enum if necessary
        if isinstance(cross_val_mode, str):
            self.cross_val_mode = CrossValidationMode(cross_val_mode.lower())
        else:
            self.cross_val_mode = cross_val_mode
        
        # Load frames and labels
        self.frames, self.labels = self._load_data()
        
        # Generate positions for sliding windows
        self.positions = self._generate_xy_positions()
        
        # Split positions for train/val based on cross-validation mode
        self.train_positions, self.val_positions = self._split_train_val()
        
        # Use appropriate positions based on train flag
        self.positions_to_use = self.train_positions if train else self.val_positions
        
        # Create temporal sequence indices
        self.sequences = self._create_sequences()
        
    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load frames and labels from the data directory.
        
        Returns:
            Tuple of (frames, labels)
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
            
        print(f"Found {len(frame_files)} frame files and {len(label_files)} label files")
        
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
    
    def _generate_xy_positions(self) -> np.ndarray:
        """
        Generate positions for sliding windows.
        
        Returns:
            Array of (x, y) positions
        """
        # Get dimensions from the first frame
        height, width = self.frames[0].shape
        
        # Generate positions - ensure we go all the way to the edge minus window size
        xpos_vec = np.arange(0, width - self.window_size + 1, self.window_step)
        ypos_vec = np.arange(0, height - self.window_size + 1, self.window_step)
        
        # Debug output
        print(f"\nWindow position generation:")
        print(f"Image dimensions: {width}x{height}")
        print(f"Window size: {self.window_size}, Window step: {self.window_step}")
        print(f"X positions: {len(xpos_vec)} values from {xpos_vec[0]} to {xpos_vec[-1]}")
        print(f"Y positions: {len(ypos_vec)} values from {ypos_vec[0]} to {ypos_vec[-1]}")
        
        # Create mesh grid
        xv, yv = np.meshgrid(xpos_vec, ypos_vec)
        positions = np.stack((xv.flatten(), yv.flatten()), axis=1)
        
        print(f"Total positions generated: {len(positions)}")
        return positions
    
    def _split_train_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split positions into training and validation sets based on cross-validation mode.
        
        The method divides the image into two parts:
        - A validation strip of either 25% of the image dimension or one window width, whichever is larger
        - The remaining area for training
        
        The split is made so that windows are fully contained within either the training or validation region.
        
        Returns:
            Tuple of (train_positions, val_positions)
        """
        if self.cross_val_mode == CrossValidationMode.NONE:
            # Use all positions for training, none for validation
            return self.positions, np.empty((0, 2), dtype=int)
        
        # Get dimensions from the first frame
        height, width = self.frames[0].shape
        
        # Define validation strip size (25% of the image dimension or one window width, whichever is larger)
        if self.cross_val_mode in [CrossValidationMode.TOP, CrossValidationMode.BOTTOM]:
            val_strip_size = max(int(height * 0.25), self.window_size)
        else:
            val_strip_size = max(int(width * 0.25), self.window_size)
        
        # Print position boundaries for debugging
        x_min = min(self.positions[:, 0]) if len(self.positions) > 0 else float('inf')
        x_max = max(self.positions[:, 0]) if len(self.positions) > 0 else float('-inf')
        y_min = min(self.positions[:, 1]) if len(self.positions) > 0 else float('inf')
        y_max = max(self.positions[:, 1]) if len(self.positions) > 0 else float('-inf')
        
        print(f"\nPosition boundaries:")
        print(f"X range: {x_min} to {x_max}")
        print(f"Y range: {y_min} to {y_max}")
        
        # Create masks for each position
        train_indices = []
        val_indices = []
        
        # Track region statistics
        top_count = left_count = bottom_count = right_count = 0
        
        for i, (x, y) in enumerate(self.positions):
            # Check if the window is entirely within the validation or training region
            window_end_x = x + self.window_size - 1  # Last pixel x-coordinate
            window_end_y = y + self.window_size - 1  # Last pixel y-coordinate
            
            is_val = False
            
            # Debug counters for position distribution
            if y < height * 0.25:
                top_count += 1
            if y >= height * 0.75:
                bottom_count += 1
            if x < width * 0.25:
                left_count += 1
            if x >= width * 0.75:
                right_count += 1
            
            if self.cross_val_mode == CrossValidationMode.TOP:
                # Window is fully in validation region if its bottom edge is within validation strip
                is_val = window_end_y < val_strip_size
            elif self.cross_val_mode == CrossValidationMode.BOTTOM:
                # Window is fully in validation region if its top edge starts within the bottom strip
                bottom_strip_start = height - val_strip_size
                is_val = y >= bottom_strip_start
            elif self.cross_val_mode == CrossValidationMode.LEFT:
                # Window is fully in validation region if its right edge is within validation strip
                is_val = window_end_x < val_strip_size
            elif self.cross_val_mode == CrossValidationMode.RIGHT:
                # Window is fully in validation region if its left edge starts within the right strip
                right_strip_start = width - val_strip_size
                is_val = x >= right_strip_start
            
            if is_val:
                val_indices.append(i)
            else:
                train_indices.append(i)
        
        train_positions = self.positions[train_indices]
        val_positions = self.positions[val_indices]
        
        # Print position distribution stats
        print(f"\nPosition distribution in image quadrants:")
        print(f"Top 25%: {top_count} positions ({top_count/len(self.positions):.1%})")
        print(f"Bottom 25%: {bottom_count} positions ({bottom_count/len(self.positions):.1%})")
        print(f"Left 25%: {left_count} positions ({left_count/len(self.positions):.1%})")
        print(f"Right 25%: {right_count} positions ({right_count/len(self.positions):.1%})")
        
        # Print split information
        print(f"\nCross-validation split summary:")
        print(f"Cross-validation mode: {self.cross_val_mode.value}")
        print(f"Window size: {self.window_size}, Window step: {self.window_step}")
        print(f"Image dimensions: {width}x{height}")
        print(f"Validation strip size: {val_strip_size} pixels")
        
        if self.cross_val_mode in [CrossValidationMode.TOP, CrossValidationMode.BOTTOM]:
            print(f"  - {val_strip_size/height:.1%} of height for {self.cross_val_mode.value}")
            if self.cross_val_mode == CrossValidationMode.TOP:
                print(f"  - Validation region: y=0 to y={val_strip_size-1}")
                print(f"  - For a position to be in validation: y+window_size-1 < {val_strip_size}")
                print(f"  - Maximum valid y-start for validation: {val_strip_size - self.window_size}")
            else:  # BOTTOM
                bottom_start = height - val_strip_size
                print(f"  - Validation region: y={bottom_start} to y={height-1}")
                print(f"  - For a position to be in validation: y >= {bottom_start}")
                print(f"  - Is max y ({y_max}) >= bottom_start ({bottom_start})? {y_max >= bottom_start}")
        else:  # LEFT or RIGHT
            print(f"  - {val_strip_size/width:.1%} of width for {self.cross_val_mode.value}")
            if self.cross_val_mode == CrossValidationMode.LEFT:
                print(f"  - Validation region: x=0 to x={val_strip_size-1}")
                print(f"  - For a position to be in validation: x+window_size-1 < {val_strip_size}")
                print(f"  - Maximum valid x-start for validation: {val_strip_size - self.window_size}")
            else:  # RIGHT
                right_start = width - val_strip_size
                print(f"  - Validation region: x={right_start} to x={width-1}")
                print(f"  - For a position to be in validation: x >= {right_start}")
                print(f"  - Is max x ({x_max}) >= right_start ({right_start})? {x_max >= right_start}")
        
        print(f"Training positions: {len(train_positions)}")
        print(f"Validation positions: {len(val_positions)}")
        
        if len(train_positions) + len(val_positions) > 0:
            train_ratio = len(train_positions) / (len(train_positions) + len(val_positions))
            val_ratio = len(val_positions) / (len(train_positions) + len(val_positions))
            print(f"Split ratio (train:val): {train_ratio:.2f}:{val_ratio:.2f}")
        else:
            print("WARNING: No positions found.")
        
        # Warn if either set is empty
        if len(train_positions) == 0:
            print(f"WARNING: No training positions. The validation strip may be too large or the window size too big.")
        
        if len(val_positions) == 0:
            print(f"WARNING: No validation positions. The validation strip may be too small or the window size too big.")
        
        # Print some useful diagnostic information
        if self.cross_val_mode == CrossValidationMode.BOTTOM:
            bottom_strip_start = height - val_strip_size
            print(f"  Debug info: Bottom strip starts at y={bottom_strip_start}")
            print(f"  Debug info: Position y-values range from {y_min} to {y_max}")
            print(f"  Debug info: Max position y ({y_max}) would need to be >= {bottom_strip_start} to have validation positions")
            print(f"  Debug info: There are {bottom_count} positions in the bottom 25% of the image")
            if bottom_count == 0:
                print(f"  Debug info: No positions are being generated in the bottom region at all!")
        elif self.cross_val_mode == CrossValidationMode.RIGHT:
            right_strip_start = width - val_strip_size
            print(f"  Debug info: Right strip starts at x={right_strip_start}")
            print(f"  Debug info: Position x-values range from {x_min} to {x_max}")
            print(f"  Debug info: Max position x ({x_max}) would need to be >= {right_strip_start} to have validation positions")
            print(f"  Debug info: There are {right_count} positions in the right 25% of the image")
            if right_count == 0:
                print(f"  Debug info: No positions are being generated in the right region at all!")
        
        return train_positions, val_positions
    
    def _create_sequences(self) -> List[Dict]:
        """
        Create temporal sequences of varying lengths.
        
        Returns:
            List of sequence specifications
        """
        sequences = []
        
        num_frames = len(self.frames)
        positions_to_use = self.positions_to_use
        
        # For each position
        for pos_idx, (x, y) in enumerate(positions_to_use):
            # For each possible sequence length from 1 to sequence_length
            for seq_len in range(1, self.sequence_length + 1):
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
    
    def _make_window(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extract a window from a frame at the given position.
        
        Args:
            frame: The frame to extract from
            x: X-coordinate of the top-left corner
            y: Y-coordinate of the top-left corner
            
        Returns:
            Extracted window
        """
        # Use y, x for indexing to be consistent with numpy convention
        return frame[y:y + self.window_size, x:x + self.window_size]
    
    def _apply_augmentation(self, windows: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to the stack of windows.
        
        Args:
            windows: Stack of windows to augment (seq_len, height, width)
            
        Returns:
            Augmented windows
        """
        if not self.augment:
            return windows
        
        # Choose a random augmentation
        aug_type = random.randint(0, 3)
        
        if aug_type == 0:
            # 90-degree rotation
            return np.stack([np.rot90(window) for window in windows])
        elif aug_type == 1:
            # Horizontal flip
            return np.stack([np.fliplr(window) for window in windows])
        elif aug_type == 2:
            # Both rotation and flip
            return np.stack([np.fliplr(np.rot90(window)) for window in windows])
        else:
            # No augmentation
            return windows
    
    def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        """
        Normalize the stack of windows.
        
        Args:
            windows: Stack of windows to normalize
            
        Returns:
            Normalized windows
        """
        if not self.normalize:
            return windows
        
        # Normalize each window individually
        normalized = np.zeros_like(windows, dtype=np.float32)
        for i, window in enumerate(windows):
            normalized[i] = (window - np.min(window)) / (np.max(window) - np.min(window) + 1e-8)
        
        return normalized
    
    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.
        
        Returns:
            Number of sequences
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence by index.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, label)
        """
        seq = self.sequences[idx]
        pos_idx = seq["pos_idx"]
        x, y = seq["x"], seq["y"]
        start_frame = seq["start_frame"]
        end_frame = seq["end_frame"]
        seq_len = seq["seq_len"]
        
        # Extract windows for each frame in the sequence
        windows = np.array([
            self._make_window(self.frames[i], x, y)
            for i in range(start_frame, end_frame + 1)
        ])
        
        # Get the label from the last frame
        label = self._make_window(self.labels[end_frame], x, y)
        
        # Apply augmentations
        windows = self._apply_augmentation(windows)
        
        # Apply the same augmentation to the label
        label = self._apply_augmentation(np.array([label]))[0]
        
        # Normalize
        windows = self._normalize_windows(windows)
        
        # Convert to torch tensors
        windows_tensor = torch.from_numpy(windows).float()
        label_tensor = torch.from_numpy(label).long()
        
        return windows_tensor, label_tensor

def get_data_loaders(
    data_dir: str,
    window_size: int = 64,
    window_step: int = 8,
    sequence_length: int = 5,
    cross_val_mode: Union[CrossValidationMode, str] = CrossValidationMode.TOP,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Directory containing the processed frames and labels
        window_size: Size of the window to extract
        window_step: Step size for sliding window
        sequence_length: Maximum number of frames to stack temporally
        cross_val_mode: Mode for cross-validation (TOP, LEFT, RIGHT, BOTTOM, NONE)
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for the data loaders
        augment: Whether to apply data augmentation
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = TemporalWindowDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=sequence_length,
        cross_val_mode=cross_val_mode,
        train=True,
        augment=augment,
        normalize=normalize
    )
    
    # Create validation dataset
    val_dataset = TemporalWindowDataset(
        data_dir=data_dir,
        window_size=window_size,
        window_step=window_step,
        sequence_length=sequence_length,
        cross_val_mode=cross_val_mode,
        train=False,
        augment=False,  # No augmentation for validation
        normalize=normalize
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Example of how to use the dataset and data loaders
    data_dir = "data"
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        window_size=256,
        window_step=16,
        sequence_length=5,
        cross_val_mode=CrossValidationMode.TOP,
        batch_size=8
    )
    
    # Print dataset information
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Get a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        break 
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from typing import List, Tuple, Optional, Callable, Union, Dict
import cv2

class STEMDefectDataset(Dataset):
    """
    Dataset for STEM defect analysis with variable sequence lengths and next-frame prediction.
    
    Args:
        frames_path: Path to the STEM frames
        labels_path: Path to the precomputed defect labels (if available)
        max_sequence_length: Maximum number of consecutive frames to stack as input
        transform: Optional transforms to apply to the input frames
        target_transform: Optional transforms to apply to the labels
        generate_labels: Whether to generate labels from frames using Maksov's method
        cv_fold: Cross-validation fold (0=top, 1=right, 2=bottom, 3=left, None=no CV)
        cv_ratio: Ratio of image to use for validation (0.0-0.5)
        split: 'train' or 'val'
    """
    def __init__(
        self,
        frames_path: str,
        labels_path: Optional[str] = None,
        max_sequence_length: int = 5,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        generate_labels: bool = False,
        cv_fold: Optional[int] = None,
        cv_ratio: float = 0.2,
        split: str = 'train'
    ):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.max_sequence_length = max_sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.generate_labels = generate_labels
        self.cv_fold = cv_fold
        self.cv_ratio = min(max(cv_ratio, 0.0), 0.5)  # Ensure between 0 and 0.5
        self.split = split
        
        # Load frame filenames and sort them
        self.frame_files = sorted([
            f for f in os.listdir(frames_path) 
            if f.endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))
        ])
        
        # Create all possible sequence-target pairs
        self.sequences = []
        for i in range(len(self.frame_files) - 1):  # -1 because we need at least one frame after for the target
            # For each starting point, we can have variable length sequences
            for seq_len in range(1, min(self.max_sequence_length + 1, len(self.frame_files) - i)):
                sequence = self.frame_files[i:i+seq_len]
                target_frame = self.frame_files[i+seq_len]
                self.sequences.append((sequence, target_frame))
        
        # If labels should be generated, initialize the label generator
        if generate_labels:
            from .ground_truth import DefectLabelGenerator
            self.label_generator = DefectLabelGenerator()
            
    def __len__(self) -> int:
        return len(self.sequences)
    
    def get_cv_mask(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Create cross-validation mask based on fold.
        
        Args:
            shape: (height, width) of the mask
            
        Returns:
            Binary mask where 1 = validation region, 0 = training region
        """
        h, w = shape
        mask = torch.zeros((1, h, w))
        
        # Calculate the strip width/height based on cv_ratio
        h_strip = int(h * self.cv_ratio)
        w_strip = int(w * self.cv_ratio)
        
        # Create mask based on CV fold
        if self.cv_fold == 0:  # Top
            mask[:, :h_strip, :] = 1.0
        elif self.cv_fold == 1:  # Right
            mask[:, :, w-w_strip:] = 1.0
        elif self.cv_fold == 2:  # Bottom
            mask[:, h-h_strip:, :] = 1.0
        elif self.cv_fold == 3:  # Left
            mask[:, :, :w_strip] = 1.0
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Get sequence and target file names
        sequence_files, target_file = self.sequences[idx]
        
        # Track original sequence length (for mask creation)
        original_seq_len = len(sequence_files)
        
        # Randomly select sequence length if in training mode
        if self.split == 'train':
            seq_len = random.randint(1, len(sequence_files))
            sequence_files = sequence_files[-seq_len:]  # Use most recent frames
        
        # Load frames
        frames = []
        for file in sequence_files:
            frame_path = os.path.join(self.frames_path, file)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frames.append(frame)
        
        # Convert to tensor
        frames = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        
        # Add channel dimension to each frame
        # Reshape to [T, C, H, W] for collation
        if frames_tensor.dim() == 3:  # [T, H, W]
            frames_tensor = frames_tensor.unsqueeze(1)  # [T, C, H, W]
        
        # Apply transform if specified
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        
        # Get or generate target label
        if self.generate_labels:
            # Load target frame
            target_path = os.path.join(self.frames_path, target_file)
            target_frame = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            
            # Generate label from target frame
            label = self.label_generator.generate_from_frame(target_frame)
            label_tensor = torch.from_numpy(label).float()
        else:
            # Load precomputed label
            label_file = target_file.replace('.png', '_label.png')
            label_path = os.path.join(self.labels_path, label_file)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Failed to load label file: {label_path}")
            # Normalize to [0, 1] - assuming labels are binary (0 or 255)
            label_tensor = torch.from_numpy(label).float() / 255.0
        
        # Add channel dimension to label if not present
        if label_tensor.dim() == 2:  # [H, W]
            label_tensor = label_tensor.unsqueeze(0)  # [C, H, W]
        
        # Apply target transform if specified
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        
        # Apply cross-validation mask
        if self.cv_fold is not None:
            cv_mask = self.get_cv_mask((label_tensor.shape[1], label_tensor.shape[2]))
            
            if self.split == 'val':
                # For validation, zero out everything except the CV region
                label_tensor = label_tensor * cv_mask
            else:  # train
                # For training, zero out the CV region
                label_tensor = label_tensor * (1 - cv_mask)
            
            # Debug information for validation set
            if self.split == 'val':
                num_positive = (label_tensor > 0.5).sum().item()
                if num_positive == 0:
                    print(f"Warning: No positive labels in validation region for sample {idx}")
                    print(f"Label stats - Min: {label_tensor.min():.3f}, Max: {label_tensor.max():.3f}")
                    print(f"CV fold: {self.cv_fold}, Target file: {target_file}")
        
        # Final sanity check on label values
        if not torch.all((label_tensor >= 0) & (label_tensor <= 1)):
            raise ValueError(f"Label values outside [0,1] range: min={label_tensor.min()}, max={label_tensor.max()}")
        
        # Return the current sequence length as well (useful for mask creation)
        actual_seq_len = frames_tensor.size(0)
        
        return frames_tensor, label_tensor, actual_seq_len


def variable_length_collate(batch):
    """
    Custom collate function for variable length sequences.
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of (input, target, seq_len) tuples from dataset
        
    Returns:
        Tuple of:
          - Padded inputs tensor [B, T, C, H, W]
          - Targets tensor [B, C, H, W]
          - Sequence mask tensor [B, T] (1 for valid frames, 0 for padding)
    """
    # Separate inputs, targets, and sequence lengths
    inputs, targets, seq_lengths = zip(*batch)
    
    # Get maximum sequence length in this batch
    max_seq_len = max(seq_lengths)
    
    # Get other dimensions (assuming all samples have same C, H, W)
    channels, height, width = inputs[0].size()[1:]
    
    # Create padded input tensor
    batch_size = len(inputs)
    padded_inputs = torch.zeros(batch_size, max_seq_len, channels, height, width, 
                              dtype=inputs[0].dtype, device=inputs[0].device)
    
    # Create mask for padded sequences (1 for valid frames, 0 for padding)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, 
                     device=inputs[0].device)
    
    # Fill padded tensor with actual data and set mask
    for i, input_tensor in enumerate(inputs):
        seq_len = input_tensor.size(0)
        padded_inputs[i, :seq_len] = input_tensor
        mask[i, :seq_len] = 1  # Mark valid frames
    
    # Stack targets (all targets should have same size)
    targets = torch.stack(targets, dim=0)
    
    return padded_inputs, targets, mask


def create_data_loaders(
    frames_path: str,
    labels_path: Optional[str] = None,
    max_sequence_length: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
    cv_fold: Optional[int] = None,
    cv_ratio: float = 0.2,
    generate_labels: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        frames_path: Path to the frames
        labels_path: Path to precomputed labels (if available)
        max_sequence_length: Maximum number of frames in each sequence
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        cv_fold: Cross-validation fold (0=top, 1=right, 2=bottom, 3=left, None=all folds)
        cv_ratio: Ratio of image to use for validation (0.0-0.5)
        generate_labels: Whether to generate labels using Maksov's method
        transform: Optional transforms for input data
        target_transform: Optional transforms for target data
        
    Returns:
        Dictionary with 'train' and 'val' data loaders
    """
    dataset_kwargs = {
        'frames_path': frames_path,
        'labels_path': labels_path,
        'max_sequence_length': max_sequence_length,
        'transform': transform,
        'target_transform': target_transform,
        'generate_labels': generate_labels,
        'cv_ratio': cv_ratio
    }
    
    train_dataset = STEMDefectDataset(
        **dataset_kwargs,
        cv_fold=cv_fold,
        split='train'
    )
    
    val_dataset = STEMDefectDataset(
        **dataset_kwargs,
        cv_fold=cv_fold,
        split='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate
    )
    
    return {'train': train_loader, 'val': val_loader}
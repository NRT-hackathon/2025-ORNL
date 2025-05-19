import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Optional, Dict, Any, Union

class STEMDataAugmentation:
    """
    Data augmentation for STEM images.
    
    Args:
        rotate_range: Range of rotation angles in degrees
        scale_range: Range of scale factors
        flip_prob: Probability of horizontal/vertical flip
        elastic_alpha: Alpha parameter for elastic transform (0 to disable)
        elastic_sigma: Sigma parameter for elastic transform
    """
    def __init__(
        self,
        rotate_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_prob: float = 0.5,
        elastic_alpha: float = 0.0,
        elastic_sigma: float = 10.0
    ):
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to a sequence of frames.
        
        Args:
            frames: Input frames of shape [T, H, W] or [T, C, H, W]
            
        Returns:
            Augmented frames of the same shape
        """
        # Add channel dimension if needed
        if frames.dim() == 3:
            frames = frames.unsqueeze(1)  # [T, 1, H, W]
        
        T, C, H, W = frames.shape
        
        # Generate random parameters
        angle = random.uniform(-self.rotate_range, self.rotate_range)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        flip_h = random.random() < self.flip_prob
        flip_v = random.random() < self.flip_prob
        
        # Apply transformations consistently to all frames
        augmented_frames = []
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            
            # Rotation and scaling
            if self.rotate_range > 0 or self.scale_range != (1.0, 1.0):
                # Calculate center
                center = torch.tensor([W / 2, H / 2])
                
                # Create rotation matrix
                angle_rad = angle * np.pi / 180.0
                alpha = scale * np.cos(angle_rad)
                beta = scale * np.sin(angle_rad)
                
                # Apply affine transformation
                affine_matrix = torch.tensor([
                    [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                    [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
                ], dtype=torch.float32)
                
                # Apply affine transformation
                grid = F.affine_grid(
                    affine_matrix.unsqueeze(0), 
                    size=(1, C, H, W), 
                    align_corners=False
                )
                frame = F.grid_sample(
                    frame.unsqueeze(0), 
                    grid, 
                    align_corners=False
                ).squeeze(0)
            
            # Flipping
            if flip_h:
                frame = torch.flip(frame, [2])  # Flip horizontally
            if flip_v:
                frame = torch.flip(frame, [1])  # Flip vertically
            
            # Elastic deformation
            if self.elastic_alpha > 0:
                # Generate displacement fields
                dx = self.elastic_alpha * torch.randn(H, W)
                dy = self.elastic_alpha * torch.randn(H, W)
                
                # Smooth displacement fields
                dx = self._gaussian_filter(dx, self.elastic_sigma)
                dy = self._gaussian_filter(dy, self.elastic_sigma)
                
                # Create sampling grid
                x, y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                indices = torch.stack([
                    torch.clamp(x + dx, 0, H - 1),
                    torch.clamp(y + dy, 0, W - 1)
                ], dim=0)
                
                # Apply transformation
                frame = self._sample(frame, indices)
            
            augmented_frames.append(frame)
        
        # Stack back into a sequence
        augmented_frames = torch.stack(augmented_frames, dim=0)
        
        return augmented_frames
    
    def _gaussian_filter(self, input: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian filter to tensor.
        
        Args:
            input: Input tensor
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Filtered tensor
        """
        size = int(2 * 4 * sigma + 1)
        size = max(size, 3)
        if size % 2 == 0:
            size += 1
            
        # Create 1D Gaussian kernel
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # Apply 2D separable convolution
        padding = size // 2
        input_padded = F.pad(input.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
        
        # Horizontal pass
        horizontal = F.conv2d(input_padded, kernel.view(1, 1, 1, size))
        
        # Vertical pass
        vertical = F.conv2d(horizontal, kernel.view(1, 1, size, 1))
        
        return vertical.squeeze(0).squeeze(0)
    
    def _sample(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Sample from input tensor using indices.
        
        Args:
            input: Input tensor [C, H, W]
            indices: Indices of shape [2, H, W]
            
        Returns:
            Sampled tensor [C, H, W]
        """
        C, H, W = input.shape
        x_indices, y_indices = indices
        
        # Normalize indices to [-1, 1] for grid_sample
        x_indices = 2 * x_indices / (H - 1) - 1
        y_indices = 2 * y_indices / (W - 1) - 1
        
        # Combine into grid of shape [H, W, 2]
        grid = torch.stack([y_indices, x_indices], dim=-1)
        
        # Apply grid_sample
        output = F.grid_sample(
            input.unsqueeze(0), 
            grid.unsqueeze(0), 
            mode='bilinear', 
            padding_mode='reflection', 
            align_corners=False
        )
        
        return output.squeeze(0)

def create_augmentation(config: Dict[str, Any]) -> Optional[STEMDataAugmentation]:
    """
    Create data augmentation based on configuration.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Data augmentation instance or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    return STEMDataAugmentation(
        rotate_range=config.get('rotate', 10.0),
        scale_range=tuple(config.get('scale', [0.9, 1.1])),
        flip_prob=0.5 if config.get('flip', True) else 0.0,
        elastic_alpha=10.0 if config.get('elastic', False) else 0.0,
        elastic_sigma=10.0
    )
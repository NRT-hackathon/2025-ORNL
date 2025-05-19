import numpy as np
import cv2
from typing import Tuple

class DefectLabelGenerator:
    """
    Implements Maksov's method for generating ground truth defect labels from STEM images.
    This class detects deviations from the ideal periodic lattice using FFT analysis
    with a two-threshold approach for detecting both bright and dark defects.
    """
    def __init__(self, mask_ratio: int = 10, thresh_low: float = 0.25, thresh_high: float = 0.75):
        """
        Initialize the defect label generator.
        
        Args:
            mask_ratio: Ratio of the image size to mask radius (higher = smaller mask)
            thresh_low: Low threshold value (normalized diff below this is considered a defect)
            thresh_high: High threshold value (normalized diff above this is considered a defect)
        """
        self.mask_ratio = mask_ratio
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
    
    def fft_mask(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes a square real space image and filters out a disk with radius equal to:
        1/mask_ratio * image size.
        
        Args:
            img: Input image
            
        Returns:
            Tuple of FFT transform and filtered FFT transform
        """
        # Take the fourier transform of the image
        f1 = np.fft.fft2(img)
        # Shift so that low spatial frequencies are in the center
        f2 = np.fft.fftshift(f1)
        # Copy the array and zero out the center
        f3 = f2.copy()
        
        # Create a circular mask
        l = int(img.shape[0] / self.mask_ratio)
        m = int(img.shape[0] / 2)
        y, x = np.ogrid[1:2*l+1, 1:2*l+1]
        mask = (x - l)**2 + (y - l)**2 <= l*l
        
        # Apply mask (zero out the center/low frequencies)
        f3[m-l:m+l, m-l:m+l] = f3[m-l:m+l, m-l:m+l] * (1 - mask)
        
        return f2, f3
    
    def fft_subtract(self, img: np.ndarray, f3: np.ndarray) -> np.ndarray:
        """
        Takes real space image and filtered FFT, reconstructs real space image
        and subtracts it from the original to identify locations with broken symmetry.
        
        Args:
            img: Original image
            f3: Filtered FFT
            
        Returns:
            Normalized difference image
        """
        # Reconstruct the filtered image
        reconstruction = np.real(np.fft.ifft2(np.fft.ifftshift(f3)))
        
        # Calculate absolute difference
        diff = np.abs(img - reconstruction)
        
        # Normalize the difference to [0, 1] range
        diff = diff - np.amin(diff)
        diff = diff / np.amax(diff)
        
        return diff
    
    def threshold_image(self, diff: np.ndarray) -> np.ndarray:
        # Calculate mean and standard deviation
        mean = np.mean(diff)
        std = np.std(diff)
        
        # Set thresholds at 3 standard deviations
        thresh_low_value = mean - 3 * std
        thresh_high_value = mean + 3 * std
        
        # Apply thresholds
        thresh_low = diff < thresh_low_value
        thresh_high = diff > thresh_high_value
        
        # Combine the thresholds (values either below low or above high are defects)
        thresh_combined = thresh_low | thresh_high
        
        return thresh_combined.astype(np.float32)
    
    def generate_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate defect labels for a single frame using FFT-based filtering.
        
        Args:
            frame: Input STEM image frame
            
        Returns:
            Binary mask with defects labeled as 1
        """
        # Ensure image is float for processing
        frame_float = frame.astype(np.float32)
        
        # Apply FFT masking
        _, f3 = self.fft_mask(frame_float)
        
        # Get difference using FFT subtraction
        diff = self.fft_subtract(frame_float, f3)
        
        # Apply thresholding
        binary_mask = self.threshold_image(diff)
        
        return binary_mask
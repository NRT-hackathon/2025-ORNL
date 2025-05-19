#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video processing utilities for STEM defect analysis.
Extracts frames from video files and optionally generates ground truth labels.
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict, Any
import logging

from .ground_truth import DefectLabelGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    generate_labels: bool = True,
    labels_dir: Optional[str] = None,
    frame_prefix: str = 'frame_',
    frame_format: str = '.png',
    resize_dims: Optional[Tuple[int, int]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    highpass_radius: float = 0.1
) -> Dict[str, Any]:
    """
    Extract frames from a video file and optionally generate ground truth labels.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        generate_labels: Whether to generate ground truth labels
        labels_dir: Directory to save ground truth labels (if None, use output_dir/labels)
        frame_prefix: Prefix for frame filenames
        frame_format: File format for saving frames
        resize_dims: Dimensions to resize frames to (width, height) or None to keep original size
        start_frame: First frame to extract
        end_frame: Last frame to extract (if None, process until the end of the video)
        step: Step size for frame extraction (extract every 'step' frames)
        highpass_radius: Radius for high-pass filter in DefectLabelGenerator
        
    Returns:
        Dictionary with extraction statistics
    """
    # Verify video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    if generate_labels:
        if labels_dir is None:
            labels_dir = os.path.join(os.path.dirname(output_dir), 'labels')
        os.makedirs(labels_dir, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine end frame
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    # Initialize label generator if needed
    label_generator = None
    if generate_labels:
        label_generator = DefectLabelGenerator()
    
    # Log extraction parameters
    logger.info(f"Extracting frames from: {video_path}")
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    logger.info(f"Frame range: {start_frame} to {end_frame} (step: {step})")
    logger.info(f"Output directory: {output_dir}")
    if generate_labels:
        logger.info(f"Labels directory: {labels_dir}")
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frames_extracted = 0
    frame_paths = []
    label_paths = []
    
    for frame_idx in tqdm(range(start_frame, end_frame, step), desc="Extracting frames"):
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame at index {frame_idx}. Stopping extraction.")
            break
        
        # Always convert to grayscale first
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        # Resize frame if specified
        if resize_dims is not None:
            frame_gray = cv2.resize(frame_gray, resize_dims, interpolation=cv2.INTER_AREA)
        
        # Generate frame filename
        frame_filename = f"{frame_prefix}{frame_idx:05d}{frame_format}"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Save frame (ensuring it's saved as grayscale)
        cv2.imwrite(frame_path, frame_gray)
        frame_paths.append(frame_path)
        
        # Generate and save label if needed
        if generate_labels and label_generator is not None:
            # Generate label
            label = label_generator.generate_from_frame(frame_gray)
            
            # Generate label filename
            label_filename = f"{frame_prefix}{frame_idx:05d}_label{frame_format}"
            label_path = os.path.join(labels_dir, label_filename)
            
            # Save label (scale to 0-255 range)
            cv2.imwrite(label_path, (label * 255).astype(np.uint8))
            label_paths.append(label_path)
        
        frames_extracted += 1
    
    # Release video capture
    cap.release()
    
    # Log extraction results
    logger.info(f"Extraction complete. Extracted {frames_extracted} frames.")
    logger.info(f"All frames saved as grayscale (single-channel) images.")
    
    # Return statistics
    return {
        'video_path': video_path,
        'frames_dir': output_dir,
        'labels_dir': labels_dir if generate_labels else None,
        'frames_extracted': frames_extracted,
        'frame_paths': frame_paths,
        'label_paths': label_paths,
        'video_properties': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        }
    }


def main():
    """Command-line interface for video frame extraction."""
    parser = argparse.ArgumentParser(description='Extract frames from video for STEM defect analysis')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--output', type=str, default='./data/frames', help='Output directory for frames')
    parser.add_argument('--labels', type=str, default=None, help='Output directory for labels (if None, use ./data/labels)')
    parser.add_argument('--generate_labels', action='store_true', help='Generate ground truth labels')
    parser.add_argument('--prefix', type=str, default='frame_', help='Prefix for frame filenames')
    parser.add_argument('--format', type=str, default='.png', help='File format for saving frames')
    parser.add_argument('--resize', type=str, default=None, help='Resize dimensions as WIDTHxHEIGHT (e.g., 256x256)')
    parser.add_argument('--start', type=int, default=0, help='First frame to extract')
    parser.add_argument('--end', type=int, default=None, help='Last frame to extract')
    parser.add_argument('--step', type=int, default=1, help='Extract every Nth frame')
    parser.add_argument('--highpass_radius', type=float, default=0.1, help='Radius for high-pass filter in label generation')
    args = parser.parse_args()
    
    # Parse resize dimensions if provided
    resize_dims = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize_dims = (width, height)
        except ValueError:
            logger.error(f"Invalid resize dimensions: {args.resize}. Expected format: WIDTHxHEIGHT")
            return
    
    # Extract frames
    try:
        stats = extract_frames_from_video(
            video_path=args.video,
            output_dir=args.output,
            generate_labels=args.generate_labels,
            labels_dir=args.labels,
            frame_prefix=args.prefix,
            frame_format=args.format,
            resize_dims=resize_dims,
            start_frame=args.start,
            end_frame=args.end,
            step=args.step,
            highpass_radius=args.highpass_radius
        )
        
        logger.info(f"Successfully extracted {stats['frames_extracted']} frames from {args.video}")
        if args.generate_labels:
            logger.info(f"Generated {len(stats['label_paths'])} label images")
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")


# Allow running this file directly
if __name__ == '__main__':
    main()
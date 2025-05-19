# STEM Defect Detection

This repository contains code for detecting defects in Scanning Transmission Electron Microscopy (STEM) images using various deep learning models, including 3D CNNs, Vision Transformers (ViT), and Mamba-based architectures.

## Project Structure

```
├── data/                     # Directory for input data and preprocessing
│   ├── frames/               # Input STEM images
│   ├── labels/               # Ground truth segmentation masks
│   ├── examples/             # Example data samples
│   ├── graphs/               # Data analysis graphs
│   ├── augmentation.py       # Data augmentation utilities
│   ├── dataset.py            # Dataset loading and processing
│   ├── ground_truth.py       # Ground truth label generation
│   └── video_processor.py    # Video frame extraction utilities
├── model/                    # Neural network model implementations
│   ├── cnn3d.py              # 3D Convolutional Neural Network
│   ├── mamba.py              # State Space Model (Mamba)
│   ├── mamba_cnn.py          # Optimized Mamba CNN hybrid
│   └── vit.py                # Vision Transformer
├── results/                  # Directory for saving results
│   ├── ablation/             # Ablation study results
│   ├── evaluation/           # Model evaluation results
│   └── final_models/         # Final trained models
├── output/                   # Directory for model output
├── output_videos/            # Output videos of model predictions
├── prediction_videos/        # Videos visualizing model predictions
├── report/                   # Report documentation
├── dataset.py                # Dataset loading and preprocessing
├── train.py                  # Model training script
├── train_final.py            # Final model training script
├── evaluate.py               # Model evaluation script
├── evaluate_ensemble.ipynb   # Notebook for ensemble model evaluation
├── ensemble_analysis.py      # Ensemble model analysis
├── ablation_study.py         # Parameter ablation study script
├── predict_sample.py         # Script for making predictions on sample data
├── create_test_data.py       # Script to create test data
├── generate_prediction_videos.py # Generate videos of model predictions
├── temporal_analysis.py      # Analysis of temporal data
├── visualize_evaluation.py   # Visualization of evaluation results
├── visualize_samples.py      # Script to visualize sample data
├── display_samples.ipynb     # Notebook for displaying samples
├── view_samples.ipynb        # Notebook for viewing samples
├── create_graphs.ipynb       # Notebook for creating result graphs
└── visualize_evaluations.ipynb # Notebook for visualizing evaluation results
```

## Data Format

The code expects data to be organized in the following structure:
```
data/
├── frames/     # Input STEM images (grayscale)
└── labels/     # Ground truth segmentation masks (binary)
```

Each frame and its corresponding label should have the same filename. Supported formats include .npy (NumPy arrays), .png, .jpg, .tif, and other common image formats.

## Data Preprocessing

The repository includes several data preprocessing utilities:

### Video Frame Extraction

The `video_processor.py` module provides tools for extracting frames from video files:

```bash
python -m data.video_processor --video path/to/video.mp4 --output data/frames --generate_labels
```

Key features:
- Extracts grayscale frames from video files
- Automatically generates ground truth defect labels
- Options for frame resizing, selection ranges, and extraction step size
- High-pass filtering for enhanced defect visibility

### Ground Truth Generation

The `ground_truth.py` module implements Maksov's method for generating defect labels:

- Uses FFT analysis to detect deviations from the ideal periodic lattice
- Applies two-threshold approach for detecting both bright and dark defects
- Automatically filters out the regular lattice pattern to highlight defects

### Data Augmentation

The `augmentation.py` module provides STEM-specific data augmentation:

- Rotation and scaling transformations
- Horizontal and vertical flipping
- Elastic deformations for realistic data variation
- Consistent transformations across temporal sequences

### Dataset Processing

The `dataset.py` module provides a specialized dataset class that:

- Handles variable-length temporal sequences
- Supports spatial cross-validation splits (TOP, LEFT, RIGHT, BOTTOM)
- Provides on-the-fly label generation
- Includes custom collation for batching variable-length sequences

## Running the Code

### 1. Data Preparation

If you need to create test data:

```bash
python create_test_data.py --output_dir data
```

To visualize samples:

```bash
python visualize_samples.py --data_dir data
```

Or use the Jupyter notebooks:
```bash
jupyter notebook display_samples.ipynb
```

### 2. Training Models

Train a model with default parameters:

```bash
python train.py --data_dir data --model_type cnn3d --batch_size 16 --epochs 50
```

Available model types:
- `cnn3d`: 3D Convolutional Neural Network
- `vit`: Vision Transformer
- `mamba`: Mamba-based architecture

Train the final models:

```bash
python train_final.py --data_dir data --results_dir results/final_models
```

### 3. Model Evaluation

Evaluate trained models:

```bash
python evaluate.py --data_dir data --models_dir results/final_models --results_dir results/evaluation
```

For ensemble model evaluation:

```bash
jupyter notebook evaluate_ensemble.ipynb
```

### 4. Generate Predictions

Make predictions on sample data:

```bash
python predict_sample.py --data_dir data --models_dir results/final_models --output_dir predictions
```

Generate prediction videos:

```bash
python generate_prediction_videos.py --data_dir data --models_dir results/final_models --output_dir prediction_videos
```

### 5. Analysis and Visualization

For temporal analysis:

```bash
python temporal_analysis.py --data_dir data --results_dir results
```

Visualize evaluation results:

```bash
python visualize_evaluation.py --results_dir results/evaluation
```

Or use the notebooks:
```bash
jupyter notebook create_graphs.ipynb
jupyter notebook visualize_evaluations.ipynb
```

## Model Architectures

### 3D CNN (cnn3d.py)
A 3D Convolutional Neural Network that processes temporal sequences of images to capture both spatial and temporal features.

### Vision Transformer (vit.py)
An implementation of the Vision Transformer architecture that uses self-attention mechanisms to process image data.

### Mamba-CNN Hybrid (mamba_cnn.py)
A hybrid model combining CNN features with the Mamba state space model for efficient sequence modeling.

## Dataset Processing

The code uses a windowing approach to process large images:
1. Images are divided into overlapping windows of size `window_size × window_size`
2. Windows are extracted with a step size of `window_step`
3. For temporal models, sequences of up to `sequence_length` consecutive frames are stacked
4. Cross-validation can be performed using spatial splits (TOP, LEFT, RIGHT, BOTTOM)

## Requirements

The project requires the following Python packages:
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- pandas
- OpenCV (cv2)
- tqdm
- scipy

## Acknowledgments

This project focuses on defect detection in STEM images using deep learning models with a focus on temporal information processing. 
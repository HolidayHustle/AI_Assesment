# Satellite Image Classification

A comprehensive deep learning project for classifying satellite imagery into five distinct categories using CNNs and Transfer Learning approaches.

## Project Overview

This project implements multiple deep learning models to classify satellite images into the following categories:
- **Cloudy**: Images with cloud coverage
- **Desert**: Arid and desert terrain
- **Green Area**: Vegetation and forested regions
- **Vehicles**: Areas with visible vehicles
- **Water**: Water bodies and aquatic regions

## Dataset

The project uses a satellite imagery dataset organized into five classes in the `data/` directory:
```
data/
├── cloudy/
├── desert/
├── green_area/
├── vehicles/
└── water/
```

All images are preprocessed and resized to 128×128 pixels (or 224×224 for transfer learning) and normalized using ImageNet statistics.

## Project Structure

### Notebooks

1. **Part A Base_CNN.ipynb**
   - Implements a basic CNN architecture from scratch
   - Covers data loading, preprocessing, and visualization
   - Trains and evaluates the baseline model
   - Includes performance metrics (accuracy, precision, recall, F1-score)

2. **Part A Deeper_CNN.ipynb**
   - Builds upon the base CNN with additional layers
   - Explores deeper network architectures for improved performance
   - Compares results with the baseline model

3. **Part B Transfer Learning.ipynb**
   - Implements transfer learning using pre-trained models
   - Leverages pretrained weights from torchvision
   - Fine-tunes the model for satellite image classification
   - Demonstrates improvements over custom CNN architectures

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Jupyter Notebook or JupyterLab
- Dependencies listed in `requirements.txt`

### Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

2. **Install dependencies from requirements.txt:**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- **torch** (2.11.0+cu126) - Deep learning framework with CUDA support
- **torchvision** (0.26.0+cu126) - Computer vision utilities
- **numpy** (2.4.3) - Numerical computing
- **pillow** (12.1.1) - Image processing
- **scikit-learn** (1.8.0) - Machine learning utilities and metrics
- **matplotlib** (3.10.8) - Data visualization
- **jupyter** - Interactive notebook environment
- **tensorflow** (2.21.0) - Additional deep learning support

## How to Run

### Using Jupyter Notebook

1. **Navigate to the project directory:**
   ```bash
   cd "Vision Task"
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open and run the desired notebook:**
   - `Part A Base_CNN.ipynb` - For baseline CNN implementation
   - `Part A Deeper_CNN.ipynb` - For deeper CNN architecture
   - `Part B Transfer Learning.ipynb` - For transfer learning approach

4. **Execute cells sequentially** - Each notebook should be run from top to bottom to ensure proper data loading and model training.

### Key Steps in Each Notebook

1. **Data Loading**: Load satellite images from the data directory
2. **Preprocessing**: Resize images to appropriate dimensions and apply normalization
3. **Visualization**: Display sample images from each class
4. **Model Definition**: Define the CNN or load pretrained architecture
5. **Training**: Train the model on the dataset
6. **Evaluation**: Assess performance using multiple metrics
7. **Visualization**: Plot training curves and classification results

## Hardware Requirements

- **GPU Recommended**: CUDA-enabled GPU for faster training
- **CPU Compatible**: Models can run on CPU, but training will be slower
- The notebooks include GPU availability checks

## Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions for each class
- **Recall**: Coverage of actual positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **Classification Report**: Detailed per-class performance metrics

## Notes

- Ensure the `data/` directory contains all five image subdirectories
- Images are automatically converted to RGB format during preprocessing
- The transform pipelines use ImageNet normalization statistics
- Models are trained using cross-entropy loss and adaptive optimizers

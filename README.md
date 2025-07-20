# ASL Fingerspelling Recognition System

This project contains a trained model for American Sign Language (ASL) fingerspelling recognition, based on the winning solution from the [Kaggle ASL Fingerspelling Competition](https://www.kaggle.com/competitions/asl-fingerspelling).

## 🎯 Overview

The system can recognize ASL fingerspelling from hand landmark data and convert it to text. It uses a Transformer-based model trained on the Kaggle ASL Fingerspelling dataset.

## 📁 Project Structure

```
ASL2NL/
├── weights/
│   ├── cfg_1/                    # PyTorch model weights (training)
│   │   ├── fold0/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   └── fold3/
│   └── cfg_2/                    # TFLite model (inference)
│       └── fold-1/
│           ├── model.tflite      # Main inference model
│           └── inference_args.json
├── asl_inference.py              # Main inference script
├── test_asl_model.py             # Basic model testing
├── test_asl_inference.py         # Comprehensive testing
└── asl-fingerspelling-recognition-w-tensorflow.ipynb  # Original notebook
```

## 🚀 Quick Start

### 1. Download Model Weights

The model weights are not included in this repository due to their large size. You need to download them separately:

**Option 1: Download from GitHub Releases**
- Go to: https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/releases
- Download `weights.zip` from the latest release
- Extract the `weights/` folder to your project root

**Option 2: Direct Download**
```bash
# Download weights (if you have wget or curl)
wget https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/releases/download/v0.0.1-alpha/weights.zip
unzip weights.zip
```

### 2. Setup Environment

```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
uv pip install torch tensorflow mediapipe numpy pandas pyarrow
```

### 3. Test the Model

```bash
# Run comprehensive tests
python test_asl_inference.py

# Run basic model test
python test_asl_model.py

# Test inference with dummy data
python asl_inference.py
```

### 4. Use the Model

```python
from asl_inference import ASLInference

# Initialize the model
model = ASLInference()

# For DataFrame input
import pandas as pd
landmarks_df = pd.DataFrame(...)  # Your landmarks data
output = model.predict(landmarks_df)

# For parquet file input
output, landmarks = model.predict_from_parquet('path/to/file.parquet', sequence_id)
```

## 🔧 Model Details

### Input Format
- **Shape**: `(1, 390)` - Single feature vector
- **Features**: 390 selected landmark coordinates (x, y, z coordinates from face, hands, and pose)
- **Data Type**: `float32`

### Output Format
- **Shape**: `(11, 63)` - Character predictions
- **Data Type**: `float32`

### Selected Features
The model uses 390 carefully selected landmark coordinates:
- Face landmarks (x, y, z coordinates)
- Left hand landmarks (21 points)
- Right hand landmarks (21 points)
- Pose landmarks (12 points)

## 📦 Weights Structure

After downloading and extracting the weights, you should have this structure:

```
weights/
├── cfg_1/                    # PyTorch training weights
│   ├── fold0/
│   │   ├── checkpoint_last_seed79464.pth
│   │   ├── checkpoint_last_seed981480.pth
│   │   ├── val_data_seed79464.pth
│   │   └── val_data_seed981480.pth
│   ├── fold1/                # Similar structure
│   ├── fold2/                # Similar structure
│   └── fold3/                # Similar structure
└── cfg_2/                    # TFLite inference model
    └── fold-1/
        ├── model.tflite      # Main inference model (39MB)
        ├── inference_args.json
        ├── checkpoint_last_seed502430.pth
        └── checkpoint_last_seed512376.pth
```

**Important**: The `model.tflite` file is the main model used for inference.

## 📊 Test Results

✅ **All tests passed!**
- Dependencies: TensorFlow 2.12.0, PyTorch 2.7.1, MediaPipe 0.10.21
- TFLite Model: Successfully loaded and tested
- PyTorch Weights: All checkpoints accessible
- MediaPipe: Hand landmark detection working

## 🎯 Usage Examples

### Example 1: Basic Inference

```python
from asl_inference import ASLInference
import numpy as np
import pandas as pd

# Initialize model
model = ASLInference()

# Create dummy landmarks data
landmarks_data = np.random.random((100, 390)).astype(np.float32)
landmarks_df = pd.DataFrame(landmarks_data, columns=model.selected_columns)

# Run inference
output = model.predict(landmarks_df)
print(f"Output shape: {output.shape}")
```

### Example 2: Process Parquet Data

```python
from asl_inference import ASLInference

# Initialize model
model = ASLInference()

# Process data from parquet file
output, landmarks = model.predict_from_parquet(
    'path/to/landmarks.parquet', 
    sequence_id=12345
)

if output is not None:
    print(f"Processed {len(landmarks)} frames")
    print(f"Output shape: {output.shape}")
```

## 🔍 Model Architecture

The model is based on the winning solution from the Kaggle competition:

1. **Feature Extraction**: Uses selected MediaPipe landmarks
2. **Transformer Architecture**: Processes the landmark sequence
3. **Output**: Character-level predictions for ASL fingerspelling

## 📝 Notes

- The model expects exactly 390 features as input
- Input data is automatically preprocessed to match the expected format
- The output needs to be decoded to convert to actual text (implementation depends on vocabulary)
- The model works with both single frames and sequences (automatically aggregated)

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   uv pip install torch tensorflow mediapipe numpy pandas pyarrow
   ```

2. **Model Loading Errors**
   - Ensure the weights directory structure is correct
   - Check that `model.tflite` and `inference_args.json` exist

3. **Input Shape Errors**
   - The model expects exactly 390 features
   - Use the provided preprocessing functions

### Getting Help

If you encounter issues:
1. Run `python test_asl_inference.py` to check system status
2. Verify all dependencies are installed
3. Check that the weights files are in the correct locations

### Weights Troubleshooting

**"Model not found" error:**
- Ensure you've downloaded the weights from the GitHub releases
- Check that `weights/cfg_2/fold-1/model.tflite` exists
- Verify the weights directory structure matches the expected layout

**"Args file not found" error:**
- Check that `weights/cfg_2/fold-1/inference_args.json` exists
- Re-download the weights if the file is missing

**Large file size:**
- The `model.tflite` file is ~39MB
- The entire weights directory is ~200MB
- This is normal for a trained deep learning model

## 📚 References

- [Kaggle Competition](https://www.kaggle.com/competitions/asl-fingerspelling)
- [Original Repository](https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution)
- [Model Weights](https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/releases)

## 🎉 Status

✅ **System Ready**: All components are working correctly!
- Model weights loaded successfully
- TFLite inference working
- Dependencies installed and tested
- Ready for ASL fingerspelling recognition 
#!/usr/bin/env python3
"""
Test script for ASL Fingerspelling Recognition Model
This script tests the model with the provided weights from the Kaggle competition.
"""

import os
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path

def load_tflite_model(model_path):
    """Load the TFLite model for inference."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✅ Successfully loaded TFLite model from {model_path}")
        return interpreter
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return None

def load_inference_args(args_path):
    """Load inference arguments from JSON file."""
    try:
        with open(args_path, 'r') as f:
            args = json.load(f)
        print(f"✅ Successfully loaded inference args from {args_path}")
        return args
    except Exception as e:
        print(f"❌ Failed to load inference args: {e}")
        return None

def test_model_input_output(interpreter):
    """Test the model's input and output details."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n📋 Model Details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    return input_details, output_details

def create_dummy_input(input_shape):
    """Create dummy input data for testing."""
    # Create random input data matching the expected shape
    dummy_input = np.random.random(input_shape).astype(np.float32)
    return dummy_input

def test_inference(interpreter, dummy_input):
    """Test model inference with dummy data."""
    try:
        # Set input tensor
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], dummy_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        print(f"✅ Inference successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5]}...")  # Show first 5 values
        
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def check_weights_structure():
    """Check the structure of the weights directory."""
    weights_dir = Path("weights")
    
    print("\n📁 Weights Directory Structure:")
    
    if not weights_dir.exists():
        print("❌ Weights directory not found!")
        return False
    
    # Check cfg_1 structure
    cfg1_dir = weights_dir / "cfg_1"
    if cfg1_dir.exists():
        print("✅ cfg_1 directory found")
        for fold in ["fold0", "fold1", "fold2", "fold3"]:
            fold_dir = cfg1_dir / fold
            if fold_dir.exists():
                files = list(fold_dir.glob("*.pth"))
                print(f"  📂 {fold}: {len(files)} .pth files")
            else:
                print(f"  ❌ {fold}: not found")
    else:
        print("❌ cfg_1 directory not found")
    
    # Check cfg_2 structure
    cfg2_dir = weights_dir / "cfg_2"
    if cfg2_dir.exists():
        print("✅ cfg_2 directory found")
        fold_dir = cfg2_dir / "fold-1"
        if fold_dir.exists():
            files = list(fold_dir.glob("*"))
            print(f"  📂 fold-1: {len(files)} files")
            for file in files:
                print(f"    📄 {file.name}")
        else:
            print("  ❌ fold-1: not found")
    else:
        print("❌ cfg_2 directory not found")
    
    return True

def main():
    """Main test function."""
    print("🧪 Testing ASL Fingerspelling Recognition Model")
    print("=" * 50)
    
    # Check weights structure
    check_weights_structure()
    
    # Test TFLite model
    print("\n🔧 Testing TFLite Model:")
    tflite_path = "weights/cfg_2/fold-1/model.tflite"
    args_path = "weights/cfg_2/fold-1/inference_args.json"
    
    if os.path.exists(tflite_path):
        # Load model
        interpreter = load_tflite_model(tflite_path)
        if interpreter:
            # Test input/output details
            input_details, output_details = test_model_input_output(interpreter)
            
            # Test inference
            input_shape = input_details[0]['shape']
            dummy_input = create_dummy_input(input_shape)
            test_inference(interpreter, dummy_input)
    else:
        print(f"❌ TFLite model not found at {tflite_path}")
    
    # Test inference args
    print("\n📄 Testing Inference Args:")
    if os.path.exists(args_path):
        args = load_inference_args(args_path)
        if args:
            print("Inference args content:")
            for key, value in args.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Inference args not found at {args_path}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
ASL Fingerspelling Recognition Inference Script
This script provides a practical interface for running inference with the trained model.
"""

import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ASLInference:
    """ASL Fingerspelling Recognition Inference Class"""
    
    def __init__(self, model_path="weights/cfg_2/fold-1/model.tflite", 
                 args_path="weights/cfg_2/fold-1/inference_args.json"):
        """Initialize the inference model."""
        self.model_path = model_path
        self.args_path = args_path
        
        # Load model and args
        self.interpreter = self._load_model()
        self.args = self._load_args()
        self.selected_columns = self.args.get('selected_columns', [])
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        print(f"   Selected columns: {len(self.selected_columns)}")
    
    def _load_model(self):
        """Load the TFLite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def _load_args(self):
        """Load inference arguments."""
        if not os.path.exists(self.args_path):
            raise FileNotFoundError(f"Args file not found at {self.args_path}")
        
        with open(self.args_path, 'r') as f:
            args = json.load(f)
        return args
    
    def preprocess_landmarks(self, landmarks_df):
        """Preprocess landmarks data for inference."""
        # Select only the required columns
        if self.selected_columns:
            available_columns = [col for col in self.selected_columns if col in landmarks_df.columns]
            if len(available_columns) != len(self.selected_columns):
                missing = set(self.selected_columns) - set(landmarks_df.columns)
                print(f"⚠️  Warning: Missing columns: {missing}")
            
            landmarks_df = landmarks_df[available_columns]
        
        # Convert to numpy array
        landmarks_array = landmarks_df.values.astype(np.float32)
        
        # The model expects a single feature vector of 390 dimensions
        # We need to aggregate the sequence data into a single vector
        if len(landmarks_array.shape) == 2:
            # If we have multiple frames, we need to aggregate them
            # For now, let's take the mean across frames
            landmarks_array = np.mean(landmarks_array, axis=0)
        
        # Ensure we have exactly 390 features
        if len(landmarks_array) != 390:
            print(f"⚠️  Warning: Expected 390 features, got {len(landmarks_array)}")
            # Pad or truncate to 390
            if len(landmarks_array) < 390:
                # Pad with zeros
                padding = np.zeros(390 - len(landmarks_array), dtype=np.float32)
                landmarks_array = np.concatenate([landmarks_array, padding])
            else:
                # Truncate
                landmarks_array = landmarks_array[:390]
        
        # Reshape to (1, 390) for batch inference
        landmarks_array = landmarks_array.reshape(1, 390)
        
        return landmarks_array
    
    def predict(self, landmarks_df):
        """Run inference on preprocessed landmarks."""
        # Preprocess input
        input_data = self.preprocess_landmarks(landmarks_df)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
    
    def predict_from_parquet(self, parquet_path, sequence_id):
        """Run inference on data from a parquet file."""
        try:
            # Read parquet file
            landmarks_df = pq.read_table(
                parquet_path,
                filters=[[('sequence_id', '=', sequence_id)]]
            ).to_pandas()
            
            if landmarks_df.empty:
                print(f"❌ No data found for sequence_id: {sequence_id}")
                return None
            
            print(f"✅ Loaded {len(landmarks_df)} frames for sequence_id: {sequence_id}")
            
            # Run prediction
            output = self.predict(landmarks_df)
            
            return output, landmarks_df
            
        except Exception as e:
            print(f"❌ Error loading parquet data: {e}")
            return None
    
    def decode_output(self, output):
        """Decode model output to get text prediction."""
        # This is a placeholder - you'll need to implement proper decoding
        # based on the model's output format and vocabulary
        print(f"Raw output shape: {output.shape}")
        print(f"Raw output sample: {output[0][:10]}")
        
        # For now, just return the raw output
        return output

def test_with_dummy_data():
    """Test the model with dummy data."""
    print("🧪 Testing with dummy data...")
    
    try:
        # Initialize model
        asl_model = ASLInference()
        
        # Create dummy landmarks data
        num_frames = 100
        num_features = len(asl_model.selected_columns) if asl_model.selected_columns else 390
        
        dummy_data = np.random.random((num_frames, num_features)).astype(np.float32)
        dummy_df = pd.DataFrame(dummy_data, columns=asl_model.selected_columns[:num_features])
        
        # Run inference
        output = asl_model.predict(dummy_df)
        
        # Decode output
        result = asl_model.decode_output(output)
        
        print(f"✅ Dummy inference successful!")
        print(f"   Input shape: {dummy_df.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dummy test failed: {e}")
        return False

def main():
    """Main function for testing."""
    print("🔤 ASL Fingerspelling Recognition Inference")
    print("=" * 50)
    
    # Test with dummy data
    success = test_with_dummy_data()
    
    if success:
        print("\n🎉 Model is working correctly!")
        print("\n📝 Usage examples:")
        print("   1. For parquet files:")
        print("      model = ASLInference()")
        print("      output, landmarks = model.predict_from_parquet('path/to/file.parquet', sequence_id)")
        print("   2. For DataFrame input:")
        print("      model = ASLInference()")
        print("      output = model.predict(landmarks_df)")
    else:
        print("\n⚠️  Model test failed. Please check the errors above.")

if __name__ == "__main__":
    main() 
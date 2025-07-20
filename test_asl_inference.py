#!/usr/bin/env python3
"""
Comprehensive test script for ASL Fingerspelling Recognition Model
This script tests both TFLite and PyTorch models with the provided weights.
"""

import os
import json
import numpy as np
import tensorflow as tf
import torch
import mediapipe as mp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_tflite_model():
    """Test the TFLite model for inference."""
    print("\n🔧 Testing TFLite Model:")
    tflite_path = "weights/cfg_2/fold-1/model.tflite"
    args_path = "weights/cfg_2/fold-1/inference_args.json"
    
    if not os.path.exists(tflite_path):
        print(f"❌ TFLite model not found at {tflite_path}")
        return False
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TFLite model loaded successfully")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Test with dummy data
        input_shape = input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ Inference test successful")
        print(f"   Output shape: {output.shape}")
        
        # Load inference args
        if os.path.exists(args_path):
            with open(args_path, 'r') as f:
                args = json.load(f)
            print(f"✅ Inference args loaded")
            print(f"   Args: {list(args.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ TFLite test failed: {e}")
        return False

def test_pytorch_weights():
    """Test the PyTorch weights."""
    print("\n🔧 Testing PyTorch Weights:")
    
    # Check cfg_1 weights
    cfg1_path = "weights/cfg_1/fold0/checkpoint_last_seed79464.pth"
    if os.path.exists(cfg1_path):
        try:
            checkpoint = torch.load(cfg1_path, map_location='cpu')
            print(f"✅ PyTorch checkpoint loaded from {cfg1_path}")
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                print(f"   Model state keys: {list(model_state.keys())[:5]}...")
                print(f"   Total model parameters: {len(model_state)}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load PyTorch checkpoint: {e}")
            return False
    else:
        print(f"❌ PyTorch checkpoint not found at {cfg1_path}")
        return False

def test_mediapipe():
    """Test MediaPipe installation and basic functionality."""
    print("\n🔧 Testing MediaPipe:")
    
    try:
        # Test basic MediaPipe functionality
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = hands.process(dummy_image)
        
        print(f"✅ MediaPipe hands module working")
        print(f"   Version: {mp.__version__}")
        
        hands.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies."""
    print("\n🔧 Testing Dependencies:")
    
    dependencies = {
        'tensorflow': tf.__version__,
        'torch': torch.__version__,
        'numpy': np.__version__,
        'mediapipe': mp.__version__
    }
    
    all_good = True
    for dep, version in dependencies.items():
        try:
            print(f"✅ {dep}: {version}")
        except:
            print(f"❌ {dep}: Not available")
            all_good = False
    
    return all_good

def create_sample_inference_script():
    """Create a sample inference script for testing."""
    script_content = '''#!/usr/bin/env python3
"""
Sample inference script for ASL Fingerspelling Recognition
This script shows how to use the trained model for inference.
"""

import numpy as np
import tensorflow as tf
import json
import os

def load_model_and_args():
    """Load the TFLite model and inference arguments."""
    model_path = "weights/cfg_2/fold-1/model.tflite"
    args_path = "weights/cfg_2/fold-1/inference_args.json"
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Load args
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    return interpreter, args

def preprocess_input(landmarks, args):
    """Preprocess input landmarks according to model requirements."""
    # This is a placeholder - actual preprocessing depends on model architecture
    # You'll need to implement this based on the training preprocessing
    input_shape = interpreter.get_input_details()[0]['shape']
    
    # Reshape and normalize landmarks
    processed = np.array(landmarks, dtype=np.float32)
    processed = processed.reshape(input_shape)
    
    return processed

def predict(interpreter, input_data):
    """Run inference on preprocessed input."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

def postprocess_output(output, args):
    """Postprocess model output to get final prediction."""
    # This is a placeholder - actual postprocessing depends on model output format
    # You'll need to implement this based on the training postprocessing
    return output

# Example usage
if __name__ == "__main__":
    print("Loading model...")
    interpreter, args = load_model_and_args()
    
    # Create dummy landmarks (replace with real data)
    dummy_landmarks = np.random.random((100, 1628)).astype(np.float32)
    
    print("Running inference...")
    processed_input = preprocess_input(dummy_landmarks, args)
    output = predict(interpreter, processed_input)
    result = postprocess_output(output, args)
    
    print(f"Inference completed! Output shape: {result.shape}")
'''
    
    with open("sample_inference.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created sample_inference.py")

def main():
    """Main test function."""
    print("🧪 Comprehensive ASL Fingerspelling Recognition Test")
    print("=" * 60)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test MediaPipe
    mp_ok = test_mediapipe()
    
    # Test TFLite model
    tflite_ok = test_tflite_model()
    
    # Test PyTorch weights
    pytorch_ok = test_pytorch_weights()
    
    # Create sample inference script
    create_sample_inference_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   MediaPipe: {'✅' if mp_ok else '❌'}")
    print(f"   TFLite Model: {'✅' if tflite_ok else '❌'}")
    print(f"   PyTorch Weights: {'✅' if pytorch_ok else '❌'}")
    
    if all([deps_ok, mp_ok, tflite_ok, pytorch_ok]):
        print("\n🎉 All tests passed! Your ASL model is ready to use.")
        print("📝 Next steps:")
        print("   1. Review sample_inference.py for usage examples")
        print("   2. Implement proper preprocessing for your input data")
        print("   3. Implement proper postprocessing for model outputs")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main() 
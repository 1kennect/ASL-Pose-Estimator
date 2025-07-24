# Testing Guide

This guide covers all testing utilities, procedures, and best practices for the ASL2NL system.

## 📚 Table of Contents

- [Overview](#overview)
- [Testing Utilities](#testing-utilities)
- [Manual Testing Procedures](#manual-testing-procedures)
- [Automated Testing](#automated-testing)
- [Performance Testing](#performance-testing)
- [Integration Testing](#integration-testing)
- [Troubleshooting Tests](#troubleshooting-tests)

## 🎯 Overview

The ASL2NL system includes several testing utilities to verify different components:

1. **System Testing** (`test_system.py`) - Basic camera and MediaPipe functionality
2. **Gesture Testing** (`test_gestures.py`) - Interactive gesture recognition testing
3. **Camera Testing** (`camera_test.py`) - Comprehensive camera hardware testing
4. **Setup Verification** (`setup.py`) - Dependency and configuration testing

## 🧰 Testing Utilities

### 1. System Test (`test_system.py`)

Tests basic system functionality including camera and MediaPipe integration.

#### Usage

```bash
python test_system.py
```

#### What it Tests

- **Camera Initialization**: Verifies camera can be opened and read from
- **MediaPipe Integration**: Tests hand detection functionality
- **Frame Processing**: Validates frame processing pipeline
- **Performance Metrics**: Measures FPS and processing speed

#### Expected Output

```
🎯 Testing ASL System Components...
✅ Camera opened successfully
✅ MediaPipe initialized successfully
📹 Starting test (press 'q' to quit)...

# Live camera window shows:
# - Hand landmarks when hands are detected
# - FPS counter
# - Hand detection status
# - Frame processing information

✅ Test completed successfully!
   Frames processed: 1234
   Average FPS: 28.5
   Total time: 43.2 seconds
```

#### Success Criteria

- Camera opens without errors
- MediaPipe detects hands correctly
- FPS > 15 for smooth operation
- No frame processing errors

### 2. Gesture Recognition Test (`test_gestures.py`)

Interactive testing for gesture recognition capabilities.

#### Usage

```bash
python test_gestures.py
```

#### What it Tests

- **Gesture Recognition Accuracy**: Tests recognition of specific ASL gestures
- **Confidence Scoring**: Validates confidence levels for different gestures
- **Real-time Performance**: Measures gesture recognition speed
- **Visual Feedback**: Provides immediate feedback on recognized gestures

#### Supported Test Gestures

| Gesture | Description | Expected Confidence |
|---------|-------------|-------------------|
| HELLO | Open hand, palm forward | > 0.7 |
| THANK | Flat hand from chin | > 0.7 |
| YES | Closed fist nodding | > 0.7 |
| NO | Index finger side to side | > 0.7 |
| PLEASE | Flat hand rubbing | > 0.7 |
| HELP | Open hand with thumb up | > 0.7 |
| LOVE | Hands over heart | > 0.7 |
| GOOD | Flat hand forward | > 0.7 |
| BAD | Thumbs down | > 0.7 |

#### Expected Output

```
🎯 Testing ASL Gesture Recognition System
✅ Camera opened successfully
✅ MediaPipe initialized successfully
📹 Starting gesture recognition test...
   Try these ASL signs:
   - HELLO (open hand, palm forward)
   - THANK YOU (flat hand from chin)
   [... more gestures ...]

# Live camera window shows:
# - Hand landmarks
# - Recognized gesture with confidence
# - FPS counter
# - Instructions

✅ Test completed successfully!
   Frames processed: 2156
   Gestures detected: 45
   Average FPS: 29.2
```

#### Success Criteria

- All target gestures recognized with confidence > 0.7
- No false positives for non-gesture hand positions
- Consistent recognition across multiple attempts
- Smooth real-time performance

### 3. Camera Hardware Test (`camera_test.py`)

Comprehensive testing of camera hardware capabilities.

#### Usage

```bash
python camera_test.py
```

#### What it Tests

- **Camera Detection**: Finds all available camera devices
- **Camera Capabilities**: Reports resolution, FPS, and other properties
- **Frame Quality**: Tests frame capture and quality
- **Multi-Camera Support**: Tests multiple camera indices

#### Expected Output

```
🎯 Camera Test Utility
🔍 Testing available cameras...

📹 Testing camera index 0...
✅ Camera 0: Working
   Resolution: 1280x720
   FPS: 30.0

📹 Testing camera index 1...
✅ Camera 1: Working
   Resolution: 640x480
   FPS: 30.0

📹 Testing camera index 2...
❌ Camera 2: Not available

✅ Found 2 working camera(s): [0, 1]

# Interactive camera selection and live preview
✅ Camera test completed!
```

#### Success Criteria

- At least one working camera detected
- Camera provides stable frame capture
- Resolution appropriate for hand detection (minimum 480x360)
- FPS adequate for real-time processing (minimum 15 FPS)

### 4. Setup Verification (`setup.py`)

Verifies system setup and dependencies.

#### Usage

```bash
python setup.py
```

#### What it Tests

- **Python Version**: Ensures Python 3.9+
- **Dependencies**: Verifies all required packages are installed
- **Package Versions**: Checks for compatible versions
- **Environment Configuration**: Validates .env file setup

#### Expected Output

```
🎯 ASL Transcription System Setup
========================================

1. Checking dependencies...
✅ mediapipe
✅ opencv-python
✅ numpy
✅ openai
✅ python-dotenv

✅ All dependencies are installed!

2. Setting up environment...
✅ Created .env file
📝 Please edit .env file and add your OpenAI API key

3. Setup complete!
```

#### Success Criteria

- All dependencies installed with correct versions
- .env file created successfully
- No import errors or version conflicts

## 🔬 Manual Testing Procedures

### Basic Functionality Test

1. **System Initialization**
   ```bash
   python asl_transcription_system.py
   ```
   - Verify camera window opens
   - Check transcription status shows "OFF"
   - Confirm all UI elements are visible

2. **Camera and Detection**
   - Position hands in camera view
   - Verify hand landmarks appear as green dots and lines
   - Check hands are tracked smoothly

3. **Gesture Recognition**
   - Press `t` to toggle transcription ON
   - Perform HELLO gesture
   - Verify "HELLO" appears in word buffer
   - Repeat with other gestures

4. **AI Processing**
   - Perform sequence: HELLO, THANK, YOU
   - Wait for AI processing (3+ words, 3+ seconds)
   - Verify sentence appears: "Hello, thank you."

5. **Controls Testing**
   - Test all keyboard controls:
     - `t` - Toggle transcription
     - `f` - Force process buffer
     - `c` - Clear buffer
     - `s` - Save transcript
     - `q` - Quit

### Gesture Accuracy Test

Test each supported gesture systematically:

```python
# Test script for gesture accuracy
gestures_to_test = [
    ("HELLO", "Open hand, palm forward"),
    ("THANK", "Flat hand from chin outward"),
    ("YES", "Closed fist nodding"),
    ("NO", "Index finger side to side"),
    ("PLEASE", "Flat hand rubbing on chest"),
    ("SORRY", "Closed fist on chest"),
    ("HELP", "Open hand with thumb up"),
    ("LOVE", "Hands over heart"),
    ("GOOD", "Flat hand forward"),
    ("BAD", "Thumbs down"),
    ("UNDERSTAND", "Index finger to head"),
    ("NAME", "Two fingers pointing"),
    ("WHAT", "Open hand questioning"),
    ("WHERE", "Index finger pointing"),
    ("WHO", "Index finger to face")
]

for gesture, description in gestures_to_test:
    print(f"Testing {gesture}: {description}")
    # Perform gesture 5 times
    # Record recognition rate and confidence
    # Note any issues or false positives
```

### Performance Benchmarking

1. **Frame Rate Test**
   - Run system for 5 minutes
   - Monitor FPS display
   - Record minimum, maximum, and average FPS
   - Expected: Average FPS > 20

2. **Recognition Latency**
   - Measure time from gesture to recognition
   - Expected: < 500ms for clear gestures

3. **AI Processing Time**
   - Measure time from buffer trigger to sentence output
   - Expected: < 5 seconds for OpenAI processing

4. **Memory Usage**
   - Monitor system memory during operation
   - Check for memory leaks during extended use

## 🤖 Automated Testing

### Unit Tests

Create automated tests for core functions:

```python
#!/usr/bin/env python3
"""
Unit tests for ASL2NL components
"""
import unittest
import numpy as np
from asl_gesture_recognition import recognize_asl_gesture

class TestGestureRecognition(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy landmark data
        self.valid_landmarks = [[0.5, 0.5, 0.0]] * 21
        self.invalid_landmarks = [[0.5, 0.5, 0.0]] * 10  # Too few landmarks
        
    def test_valid_landmarks(self):
        """Test with valid landmark data."""
        word, confidence = recognize_asl_gesture(self.valid_landmarks)
        self.assertIsInstance(word, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_invalid_landmarks(self):
        """Test with invalid landmark data."""
        word, confidence = recognize_asl_gesture(self.invalid_landmarks)
        self.assertEqual(word, "")
        self.assertEqual(confidence, 0.0)
    
    def test_empty_landmarks(self):
        """Test with empty landmarks."""
        word, confidence = recognize_asl_gesture([])
        self.assertEqual(word, "")
        self.assertEqual(confidence, 0.0)
    
    def test_none_landmarks(self):
        """Test with None landmarks."""
        word, confidence = recognize_asl_gesture(None)
        self.assertEqual(word, "")
        self.assertEqual(confidence, 0.0)

class TestSystemComponents(unittest.TestCase):
    
    def test_mediapipe_import(self):
        """Test MediaPipe import."""
        try:
            import mediapipe as mp
            self.assertTrue(True)
        except ImportError:
            self.fail("MediaPipe import failed")
    
    def test_opencv_import(self):
        """Test OpenCV import."""
        try:
            import cv2
            self.assertTrue(True)
        except ImportError:
            self.fail("OpenCV import failed")
    
    def test_openai_import(self):
        """Test OpenAI import."""
        try:
            import openai
            self.assertTrue(True)
        except ImportError:
            self.fail("OpenAI import failed")

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Test component interactions:

```python
#!/usr/bin/env python3
"""
Integration tests for ASL2NL system
"""
import unittest
import cv2
import numpy as np
from asl_transcription_system import ASLTranscriptionSystem

class TestSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test system."""
        self.system = ASLTranscriptionSystem(use_openai=False)  # Disable AI for testing
    
    def tearDown(self):
        """Clean up test system."""
        if self.system:
            self.system.cleanup()
    
    def test_system_initialization(self):
        """Test system initializes correctly."""
        self.assertIsNotNone(self.system)
        self.assertFalse(self.system.transcription_active)
        self.assertEqual(len(self.system.word_buffer), 0)
    
    def test_camera_fallback(self):
        """Test camera fallback mechanism."""
        # This test depends on available cameras
        result = self.system.start_camera(999)  # Non-existent camera
        # Should fallback to available cameras
        self.assertIsInstance(result, bool)
    
    def test_frame_processing(self):
        """Test frame processing pipeline."""
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame (should not crash)
        try:
            result_frame = self.system.process_frame(test_frame)
            self.assertEqual(result_frame.shape, test_frame.shape)
        except Exception as e:
            self.fail(f"Frame processing failed: {e}")

if __name__ == '__main__':
    unittest.main()
```

### Running Automated Tests

```bash
# Run unit tests
python -m unittest test_unit.py -v

# Run integration tests
python -m unittest test_integration.py -v

# Run all tests
python -m unittest discover tests/ -v
```

## 📊 Performance Testing

### Benchmark Script

```python
#!/usr/bin/env python3
"""
Performance benchmark for ASL2NL system
"""
import time
import cv2
import statistics
from asl_transcription_system import ASLTranscriptionSystem

class PerformanceBenchmark:
    
    def __init__(self):
        self.system = ASLTranscriptionSystem(use_openai=False)
        self.metrics = {
            'fps': [],
            'frame_processing_time': [],
            'gesture_recognition_time': [],
            'memory_usage': []
        }
    
    def run_benchmark(self, duration_seconds=60):
        """Run performance benchmark."""
        print(f"🏃 Running {duration_seconds}s performance benchmark...")
        
        if not self.system.start_camera():
            print("❌ Cannot start camera for benchmark")
            return
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            # Capture frame
            ret, frame = self.system.cap.read()
            if not ret:
                continue
            
            # Process frame
            processed_frame = self.system.process_frame(frame)
            
            frame_end = time.time()
            frame_processing_time = frame_end - frame_start
            
            # Record metrics
            self.metrics['frame_processing_time'].append(frame_processing_time)
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                self.metrics['fps'].append(current_fps)
        
        self.system.cleanup()
        self.generate_report()
    
    def generate_report(self):
        """Generate performance report."""
        print("\n📊 PERFORMANCE REPORT")
        print("=" * 40)
        
        if self.metrics['fps']:
            avg_fps = statistics.mean(self.metrics['fps'])
            min_fps = min(self.metrics['fps'])
            max_fps = max(self.metrics['fps'])
            
            print(f"FPS Statistics:")
            print(f"  Average: {avg_fps:.2f}")
            print(f"  Minimum: {min_fps:.2f}")
            print(f"  Maximum: {max_fps:.2f}")
        
        if self.metrics['frame_processing_time']:
            avg_processing = statistics.mean(self.metrics['frame_processing_time']) * 1000
            max_processing = max(self.metrics['frame_processing_time']) * 1000
            
            print(f"\nFrame Processing Time:")
            print(f"  Average: {avg_processing:.2f}ms")
            print(f"  Maximum: {max_processing:.2f}ms")
        
        # Performance assessment
        if self.metrics['fps']:
            avg_fps = statistics.mean(self.metrics['fps'])
            if avg_fps >= 25:
                print(f"\n✅ Performance: EXCELLENT ({avg_fps:.1f} FPS)")
            elif avg_fps >= 20:
                print(f"\n✅ Performance: GOOD ({avg_fps:.1f} FPS)")
            elif avg_fps >= 15:
                print(f"\n⚠️  Performance: ACCEPTABLE ({avg_fps:.1f} FPS)")
            else:
                print(f"\n❌ Performance: POOR ({avg_fps:.1f} FPS)")

if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    benchmark.run_benchmark(60)  # 60 second benchmark
```

### Expected Performance Metrics

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Average FPS | ≥25 | ≥20 | ≥15 | <15 |
| Frame Processing | <30ms | <40ms | <50ms | ≥50ms |
| Gesture Recognition | <100ms | <200ms | <300ms | ≥300ms |
| Memory Usage | <500MB | <750MB | <1GB | ≥1GB |

## 🔗 Integration Testing

### OpenAI API Test

```python
#!/usr/bin/env python3
"""
OpenAI API integration test
"""
import os
import openai
from dotenv import load_dotenv

def test_openai_integration():
    """Test OpenAI API integration."""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Test basic completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an ASL interpreter."},
                {"role": "user", "content": "Convert: HELLO THANK YOU"}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI API test successful")
        print(f"   Input: HELLO THANK YOU")
        print(f"   Output: {result}")
        return True
    
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

if __name__ == '__main__':
    test_openai_integration()
```

### Camera Integration Test

```python
#!/usr/bin/env python3
"""
Camera integration test with MediaPipe
"""
import cv2
import mediapipe as mp

def test_camera_mediapipe_integration():
    """Test camera and MediaPipe integration."""
    print("🧪 Testing Camera + MediaPipe Integration")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    # Test camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No camera available")
        return False
    
    print("✅ Camera and MediaPipe initialized")
    
    # Process test frames
    test_frames = 100
    successful_frames = 0
    hand_detections = 0
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        successful_frames += 1
        
        if results.multi_hand_landmarks:
            hand_detections += 1
    
    cap.release()
    
    success_rate = (successful_frames / test_frames) * 100
    detection_rate = (hand_detections / successful_frames) * 100 if successful_frames > 0 else 0
    
    print(f"✅ Integration test completed")
    print(f"   Frame success rate: {success_rate:.1f}%")
    print(f"   Hand detection rate: {detection_rate:.1f}%")
    
    return success_rate > 95  # 95% success rate required

if __name__ == '__main__':
    test_camera_mediapipe_integration()
```

## 🔧 Troubleshooting Tests

### Diagnostic Test Suite

```python
#!/usr/bin/env python3
"""
Comprehensive diagnostic test suite
"""
import sys
import os
import cv2
import importlib

class DiagnosticTests:
    
    def __init__(self):
        self.results = {}
    
    def run_all_diagnostics(self):
        """Run all diagnostic tests."""
        tests = [
            ("Python Version", self.test_python_version),
            ("Required Imports", self.test_imports),
            ("Camera Hardware", self.test_camera_hardware),
            ("MediaPipe", self.test_mediapipe),
            ("OpenAI Configuration", self.test_openai_config),
            ("File Permissions", self.test_file_permissions),
            ("Environment Variables", self.test_environment)
        ]
        
        print("🔍 Running Diagnostic Tests")
        print("=" * 40)
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}...")
            try:
                result = test_func()
                self.results[test_name] = result
                if result['status'] == 'pass':
                    print(f"   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED: {result['message']}")
            except Exception as e:
                self.results[test_name] = {'status': 'error', 'message': str(e)}
                print(f"   💥 ERROR: {e}")
        
        self.generate_diagnostic_report()
    
    def test_python_version(self):
        """Test Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            return {'status': 'pass', 'version': f"{version.major}.{version.minor}"}
        else:
            return {'status': 'fail', 'message': f"Python {version.major}.{version.minor} < 3.9"}
    
    def test_imports(self):
        """Test all required imports."""
        required_modules = [
            'cv2', 'mediapipe', 'numpy', 'openai', 'dotenv'
        ]
        
        failed_imports = []
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                failed_imports.append(module)
        
        if failed_imports:
            return {'status': 'fail', 'message': f"Missing: {', '.join(failed_imports)}"}
        else:
            return {'status': 'pass'}
    
    def test_camera_hardware(self):
        """Test camera hardware availability."""
        working_cameras = []
        
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    working_cameras.append(i)
                cap.release()
        
        if working_cameras:
            return {'status': 'pass', 'cameras': working_cameras}
        else:
            return {'status': 'fail', 'message': 'No working cameras found'}
    
    def test_mediapipe(self):
        """Test MediaPipe functionality."""
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands()
            return {'status': 'pass', 'version': mp.__version__}
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}
    
    def test_openai_config(self):
        """Test OpenAI configuration."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {'status': 'fail', 'message': 'OPENAI_API_KEY not set'}
        elif api_key == 'your_openai_api_key_here':
            return {'status': 'fail', 'message': 'OPENAI_API_KEY not configured'}
        else:
            return {'status': 'pass'}
    
    def test_file_permissions(self):
        """Test file system permissions."""
        try:
            # Test write permissions
            test_file = 'test_permissions.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return {'status': 'pass'}
        except Exception as e:
            return {'status': 'fail', 'message': f'Write permission error: {e}'}
    
    def test_environment(self):
        """Test environment configuration."""
        issues = []
        
        # Check .env file
        if not os.path.exists('.env'):
            issues.append('.env file missing')
        
        # Check required directories
        if not os.path.exists('docs'):
            issues.append('docs directory missing')
        
        if issues:
            return {'status': 'fail', 'message': '; '.join(issues)}
        else:
            return {'status': 'pass'}
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        print("\n" + "=" * 40)
        print("📋 DIAGNOSTIC REPORT")
        print("=" * 40)
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'pass')
        total = len(self.results)
        
        print(f"Overall Status: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All diagnostics passed - system ready!")
        else:
            print("\n❌ Issues found:")
            for test_name, result in self.results.items():
                if result['status'] != 'pass':
                    print(f"   • {test_name}: {result.get('message', 'Unknown error')}")
            
            print("\n💡 Recommended actions:")
            self.generate_recommendations()
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        for test_name, result in self.results.items():
            if result['status'] != 'pass':
                if test_name == "Python Version":
                    print("   • Upgrade to Python 3.9 or higher")
                elif test_name == "Required Imports":
                    print("   • Run: pip install -r requirements.txt")
                elif test_name == "Camera Hardware":
                    print("   • Check camera connections and permissions")
                elif test_name == "OpenAI Configuration":
                    print("   • Set OPENAI_API_KEY in .env file")
                elif test_name == "File Permissions":
                    print("   • Check directory write permissions")

if __name__ == '__main__':
    diagnostics = DiagnosticTests()
    diagnostics.run_all_diagnostics()
```

## 📈 Test Reports and Metrics

### Test Execution

Run comprehensive test suite:

```bash
# Basic functionality tests
python test_system.py
python test_gestures.py
python camera_test.py

# Automated tests
python -m unittest discover tests/ -v

# Performance benchmarks
python benchmark.py

# Diagnostic tests
python diagnostics.py
```

### Success Criteria Summary

| Component | Test | Success Criteria |
|-----------|------|------------------|
| **System** | Basic functionality | Camera opens, MediaPipe works, FPS > 15 |
| **Gestures** | Recognition accuracy | >80% recognition rate for supported gestures |
| **Camera** | Hardware compatibility | At least one working camera with 480p+ |
| **Performance** | Real-time processing | Average FPS > 20, processing time < 50ms |
| **Integration** | API connectivity | OpenAI API responds within 5 seconds |
| **Setup** | Dependencies | All packages installed, no import errors |

### Continuous Testing

For ongoing development, implement automated testing:

```bash
#!/bin/bash
# test_runner.sh - Automated test runner

echo "🚀 Running ASL2NL Test Suite"

# Run unit tests
echo "📋 Unit Tests..."
python -m unittest discover tests/unit/ -v

# Run integration tests
echo "🔗 Integration Tests..."
python -m unittest discover tests/integration/ -v

# Run performance benchmarks
echo "⚡ Performance Tests..."
python tests/benchmark.py

# Generate test report
echo "📊 Generating Test Report..."
python tests/generate_report.py

echo "✅ Test suite completed!"
```

This comprehensive testing guide ensures the ASL2NL system works reliably across different environments and use cases.
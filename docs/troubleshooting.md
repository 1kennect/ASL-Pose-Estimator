# Troubleshooting Guide

This guide covers common issues, solutions, and diagnostic procedures for the ASL2NL system.

## 📚 Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Camera Problems](#camera-problems)
- [Gesture Recognition Issues](#gesture-recognition-issues)
- [AI Processing Problems](#ai-processing-problems)
- [Performance Issues](#performance-issues)
- [Environment Configuration](#environment-configuration)
- [Advanced Troubleshooting](#advanced-troubleshooting)

## 🔍 Quick Diagnostics

### Run System Diagnostics

First, run the built-in diagnostic tool:

```bash
python setup.py
```

This will check:
- Python version compatibility
- Required dependencies
- Environment configuration
- Basic system functionality

### Check System Status

```bash
# Test individual components
python test_system.py      # Camera and MediaPipe
python test_gestures.py    # Gesture recognition
python camera_test.py      # Camera hardware
```

## 🛠 Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
- `pip install -r requirements.txt` fails
- Import errors when running the system
- Missing module errors

**Solutions:**

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install packages individually:**
   ```bash
   pip install mediapipe>=0.10.0
   pip install opencv-python>=4.8.0
   pip install numpy>=1.21.0
   pip install openai>=1.0.0
   pip install python-dotenv>=1.0.0
   ```

4. **Platform-specific issues:**
   
   **macOS:**
   ```bash
   # If you get permission errors
   pip install --user -r requirements.txt
   
   # For Apple Silicon Macs
   pip install tensorflow-macos
   ```
   
   **Windows:**
   ```bash
   # Use conda if pip fails
   conda install -c conda-forge mediapipe opencv
   pip install openai python-dotenv
   ```
   
   **Linux:**
   ```bash
   # Install system dependencies first
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   pip install -r requirements.txt
   ```

### Problem: Python Version Issues

**Symptoms:**
- "Python 3.9+ required" error
- Syntax errors in code
- Feature not available errors

**Solutions:**

1. **Check Python version:**
   ```bash
   python --version
   python3 --version
   ```

2. **Install Python 3.9+:**
   - **Windows/macOS:** Download from [python.org](https://python.org)
   - **Linux:** 
     ```bash
     sudo apt-get install python3.9 python3.9-pip
     # or
     sudo yum install python39 python39-pip
     ```

3. **Use specific Python version:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   python3.9 -m pip install -r requirements.txt
   ```

## 📹 Camera Problems

### Problem: No Camera Detected

**Symptoms:**
- "No working camera found" error
- Camera window doesn't open
- Black screen in camera window

**Solutions:**

1. **Check camera availability:**
   ```bash
   python camera_test.py
   ```

2. **Try different camera indices:**
   ```bash
   # Edit .env file
   CAMERA_INDEX=0  # Try 0, 1, 2, etc.
   ```

3. **Check camera permissions:**
   
   **macOS:**
   - System Preferences → Security & Privacy → Camera
   - Enable camera access for Terminal/Python
   
   **Windows:**
   - Settings → Privacy → Camera
   - Enable camera access for apps
   
   **Linux:**
   ```bash
   # Check if camera is recognized
   lsusb | grep -i camera
   ls /dev/video*
   
   # Add user to video group
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

4. **Test camera externally:**
   ```bash
   # Linux
   ffplay /dev/video0
   
   # macOS
   # Use Photo Booth or QuickTime
   
   # Windows
   # Use Camera app
   ```

### Problem: Camera Opens but No Video

**Symptoms:**
- Camera initializes successfully
- Window opens but shows black/frozen frame
- "Could not read frame" errors

**Solutions:**

1. **Check camera format:**
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
   print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
   print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
   cap.release()
   ```

2. **Set camera properties:**
   ```python
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   cap.set(cv2.CAP_PROP_FPS, 30)
   ```

3. **Try different backends:**
   ```python
   # DirectShow (Windows)
   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
   
   # V4L2 (Linux)
   cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
   
   # AVFoundation (macOS)
   cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
   ```

4. **Close other camera applications:**
   - Close Zoom, Skype, Teams, etc.
   - Check background processes using camera

### Problem: Poor Camera Performance

**Symptoms:**
- Low FPS (< 15)
- Laggy camera feed
- Frame drops

**Solutions:**

1. **Reduce camera resolution:**
   ```python
   # In asl_transcription_system.py, modify start_camera()
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. **Optimize processing:**
   ```python
   # Reduce MediaPipe accuracy for speed
   hands = mp_hands.Hands(
       min_detection_confidence=0.5,  # Lower from 0.7
       min_tracking_confidence=0.3    # Lower from 0.5
   )
   ```

3. **Check system resources:**
   ```bash
   # Linux/macOS
   top
   htop
   
   # Windows
   # Task Manager → Performance
   ```

## 👋 Gesture Recognition Issues

### Problem: Gestures Not Recognized

**Symptoms:**
- Hand landmarks visible but no gestures detected
- Low confidence scores (< 0.6)
- Wrong gestures recognized

**Solutions:**

1. **Check hand visibility:**
   - Ensure hands are clearly visible
   - Good lighting conditions
   - Plain background behind hands
   - Hands within camera frame

2. **Adjust confidence threshold:**
   ```python
   # In asl_transcription_system.py, modify process_frame()
   if word and confidence > 0.5:  # Lower from 0.6
       self.word_buffer.append(word)
   ```

3. **Test gesture recognition:**
   ```bash
   python test_gestures.py
   ```

4. **Improve gesture technique:**
   - Hold gestures for 1-2 seconds
   - Make clear, deliberate movements
   - Practice consistent hand positions
   - Refer to gesture descriptions in documentation

### Problem: False Positive Recognition

**Symptoms:**
- Random gestures detected during normal hand movement
- Incorrect gestures recognized
- Buffer fills with unwanted words

**Solutions:**

1. **Increase confidence threshold:**
   ```python
   if word and confidence > 0.8:  # Increase from 0.6
       self.word_buffer.append(word)
   ```

2. **Add gesture debouncing:**
   ```python
   # In ASLTranscriptionSystem class
   self.last_gesture_time = 0
   self.gesture_debounce = 1.0  # 1 second between gestures
   
   # In process_frame()
   current_time = time.time()
   if (word and confidence > 0.6 and 
       current_time - self.last_gesture_time > self.gesture_debounce):
       self.word_buffer.append(word)
       self.last_gesture_time = current_time
   ```

3. **Improve gesture specificity:**
   - Be more deliberate with gestures
   - Pause between different gestures
   - Keep hands still when not signing

### Problem: Inconsistent Recognition

**Symptoms:**
- Same gesture sometimes recognized, sometimes not
- Varying confidence scores for identical gestures
- Recognition depends on hand position/angle

**Solutions:**

1. **Consistent gesture positioning:**
   - Keep hands at consistent distance from camera
   - Maintain similar hand angles
   - Use consistent lighting

2. **Calibrate gesture recognition:**
   ```python
   # Create custom recognition thresholds
   def improved_recognize_gesture(landmarks):
       word, confidence = recognize_asl_gesture(landmarks)
       
       # Apply custom thresholds per gesture
       gesture_thresholds = {
           'HELLO': 0.7,
           'THANK': 0.8,
           'YES': 0.6,
           # ... add more
       }
       
       threshold = gesture_thresholds.get(word, 0.6)
       if confidence < threshold:
           return "", 0.0
       return word, confidence
   ```

3. **Add gesture validation:**
   ```python
   # Require gesture to be detected multiple times
   gesture_history = []
   
   if word and confidence > 0.6:
       gesture_history.append(word)
       if len(gesture_history) > 5:
           gesture_history.pop(0)
       
       # Only accept if gesture appears 3+ times in last 5 detections
       if gesture_history.count(word) >= 3:
           self.word_buffer.append(word)
   ```

## 🤖 AI Processing Problems

### Problem: OpenAI API Errors

**Symptoms:**
- "OpenAI API key not found" error
- "Invalid API key" error
- "Rate limit exceeded" error
- "Network connection error"

**Solutions:**

1. **Check API key configuration:**
   ```bash
   # Verify .env file exists and contains:
   cat .env
   # Should show: OPENAI_API_KEY=sk-...
   ```

2. **Validate API key:**
   ```python
   import openai
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
   
   # Test API call
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello"}],
       max_tokens=5
   )
   print(response.choices[0].message.content)
   ```

3. **Handle rate limits:**
   ```python
   import time
   import random
   
   def call_openai_with_retry(client, messages, max_retries=3):
       for attempt in range(max_retries):
           try:
               response = client.chat.completions.create(
                   model="gpt-3.5-turbo",
                   messages=messages,
                   max_tokens=50
               )
               return response.choices[0].message.content
           except openai.RateLimitError:
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               print(f"Rate limit hit, waiting {wait_time:.1f}s...")
               time.sleep(wait_time)
           except Exception as e:
               print(f"API error: {e}")
               return None
       return None
   ```

4. **Check account status:**
   - Visit [OpenAI Platform](https://platform.openai.com)
   - Check usage limits and billing
   - Verify API key permissions

### Problem: Poor AI Sentence Quality

**Symptoms:**
- AI returns "INSUFFICIENT_CONTEXT" frequently
- Generated sentences don't make sense
- AI doesn't understand ASL word sequences

**Solutions:**

1. **Improve AI prompts:**
   ```python
   def better_construct_sentence(self, word_sequence):
       prompt = f"""
   You are an expert ASL (American Sign Language) interpreter. 
   Convert this sequence of ASL signs into natural English:
   
   ASL Signs: {word_sequence}
   
   Rules:
   - ASL has different grammar than English
   - Consider context and common ASL phrases
   - If unclear, provide the most likely interpretation
   - Only respond with the English sentence, nothing else
   
   English sentence:"""
       
       response = self.openai_client.chat.completions.create(
           model="gpt-3.5-turbo",
           messages=[{"role": "user", "content": prompt}],
           max_tokens=50,
           temperature=0.1  # Lower temperature for more consistent results
       )
       return response.choices[0].message.content.strip()
   ```

2. **Add context awareness:**
   ```python
   def construct_with_context(self, word_sequence):
       # Include recent sentences for context
       context = "\n".join(self.sentence_history[-3:])
       
       prompt = f"""
   Previous context: {context}
   
   New ASL signs: {word_sequence}
   
   Convert to English considering the context:"""
       # ... rest of API call
   ```

3. **Use rule-based fallback:**
   ```python
   def hybrid_sentence_construction(self, word_sequence):
       # Try AI first
       ai_result = self._construct_sentence_with_ai(word_sequence)
       
       if ai_result and ai_result not in ["INSUFFICIENT_CONTEXT", "RANDOM_LETTERS"]:
           return ai_result
       
       # Fallback to rule-based
       return self._rule_based_construction(word_sequence)
   
   def _rule_based_construction(self, word_sequence):
       words = word_sequence.split()
       
       # Simple patterns
       if len(words) == 1:
           return words[0].lower()
       elif len(words) == 2 and words[1] == "YOU":
           return f"{words[0].lower()} you"
       # Add more patterns...
       else:
           return " ".join(words).lower()
   ```

### Problem: Slow AI Processing

**Symptoms:**
- Long delays between gesture recognition and sentence output
- System feels unresponsive
- Timeout errors

**Solutions:**

1. **Optimize API calls:**
   ```python
   # Reduce max_tokens for faster response
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=messages,
       max_tokens=30,  # Reduce from 50
       temperature=0.1
   )
   ```

2. **Use faster model:**
   ```python
   # In .env file
   OPENAI_MODEL=gpt-3.5-turbo-instruct  # Faster than chat models
   ```

3. **Implement caching:**
   ```python
   sentence_cache = {}
   
   def cached_ai_processing(self, word_sequence):
       if word_sequence in sentence_cache:
           return sentence_cache[word_sequence]
       
       result = self._construct_sentence_with_ai(word_sequence)
       sentence_cache[word_sequence] = result
       return result
   ```

4. **Reduce processing frequency:**
   ```python
   # Increase minimum interval between AI calls
   self.ai_process_interval = 5.0  # Increase from 3.0
   ```

## ⚡ Performance Issues

### Problem: Low Frame Rate

**Symptoms:**
- FPS below 15
- Jerky camera movement
- Delayed response to gestures

**Solutions:**

1. **Optimize MediaPipe settings:**
   ```python
   hands = mp_hands.Hands(
       static_image_mode=False,
       max_num_hands=1,           # Reduce from 2
       min_detection_confidence=0.5,  # Reduce from 0.7
       min_tracking_confidence=0.3    # Reduce from 0.5
   )
   ```

2. **Reduce processing frequency:**
   ```python
   # Process every nth frame
   self.frame_skip = 2  # Process every 2nd frame
   
   if self.frame_count % self.frame_skip == 0:
       # Process frame for gestures
       pass
   ```

3. **Optimize gesture recognition:**
   ```python
   # Skip gesture recognition when transcription is off
   if self.transcription_active and results.multi_hand_landmarks:
       # Only process when needed
       word, confidence = self._recognize_asl_gesture(hand_landmarks)
   ```

4. **System optimization:**
   - Close unnecessary applications
   - Use dedicated GPU if available
   - Increase system RAM
   - Use SSD instead of HDD

### Problem: High Memory Usage

**Symptoms:**
- System becomes slow over time
- Memory usage continuously increases
- Out of memory errors

**Solutions:**

1. **Limit buffer sizes:**
   ```python
   # Reduce buffer sizes
   self.word_buffer = deque(maxlen=5)      # Reduce from 10
   self.sentence_history = self.sentence_history[-10:]  # Keep only last 10
   ```

2. **Clean up resources:**
   ```python
   def cleanup_resources(self):
       """Enhanced cleanup with memory management."""
       if self.cap:
           self.cap.release()
       cv2.destroyAllWindows()
       
       # Clear buffers
       self.word_buffer.clear()
       self.sentence_history.clear()
       
       # Force garbage collection
       import gc
       gc.collect()
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   
   def log_memory_usage(self):
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Memory usage: {memory_mb:.1f} MB")
   ```

## ⚙️ Environment Configuration

### Problem: .env File Issues

**Symptoms:**
- Environment variables not loaded
- "OPENAI_API_KEY not found" despite setting it
- Configuration changes not taking effect

**Solutions:**

1. **Verify .env file location:**
   ```bash
   # Should be in project root directory
   ls -la .env
   ```

2. **Check .env file format:**
   ```bash
   # Correct format (no spaces around =)
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-3.5-turbo
   CAMERA_INDEX=0
   
   # Incorrect format
   OPENAI_API_KEY = sk-your-key-here  # No spaces!
   ```

3. **Test environment loading:**
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   print(f"API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
   print(f"API Key starts with: {os.getenv('OPENAI_API_KEY', '')[:10]}...")
   ```

4. **Force reload environment:**
   ```python
   load_dotenv(override=True)  # Override existing environment variables
   ```

### Problem: Path and Import Issues

**Symptoms:**
- "Module not found" errors
- Import errors for local modules
- File not found errors

**Solutions:**

1. **Check working directory:**
   ```python
   import os
   print(f"Current directory: {os.getcwd()}")
   print(f"Files in directory: {os.listdir('.')}")
   ```

2. **Add project to Python path:**
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.abspath(__file__)))
   ```

3. **Use absolute imports:**
   ```python
   # Instead of
   from asl_gesture_recognition import recognize_asl_gesture
   
   # Use
   from .asl_gesture_recognition import recognize_asl_gesture
   ```

## 🔧 Advanced Troubleshooting

### Debug Mode

Enable detailed logging for troubleshooting:

```python
# Add to asl_transcription_system.py
DEBUG = True  # Set to True for debugging

def debug_print(message):
    if DEBUG:
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] DEBUG: {message}")

# Use throughout code
debug_print(f"Camera started with index: {camera_index}")
debug_print(f"Gesture recognized: {word} ({confidence})")
debug_print(f"Word buffer: {list(self.word_buffer)}")
```

### System Information Collection

Create a diagnostic report:

```python
#!/usr/bin/env python3
"""
System information collector for troubleshooting
"""
import sys
import os
import platform
import cv2
import subprocess

def collect_system_info():
    """Collect comprehensive system information."""
    info = {
        "System": {
            "OS": platform.system(),
            "Version": platform.version(),
            "Architecture": platform.architecture(),
            "Python": sys.version,
            "Working Directory": os.getcwd()
        },
        "Environment": {
            "PATH": os.environ.get("PATH", "Not set"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
            "Virtual Environment": os.environ.get("VIRTUAL_ENV", "Not active")
        },
        "Dependencies": {},
        "Hardware": {}
    }
    
    # Check dependencies
    dependencies = ["cv2", "mediapipe", "numpy", "openai", "dotenv"]
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "Unknown")
            info["Dependencies"][dep] = version
        except ImportError:
            info["Dependencies"][dep] = "Not installed"
    
    # Check cameras
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    info["Hardware"]["Cameras"] = cameras
    
    # Check GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        info["Hardware"]["GPU"] = "NVIDIA GPU detected" if result.returncode == 0 else "No NVIDIA GPU"
    except:
        info["Hardware"]["GPU"] = "Unknown"
    
    return info

def generate_report():
    """Generate diagnostic report."""
    info = collect_system_info()
    
    print("🔍 SYSTEM DIAGNOSTIC REPORT")
    print("=" * 50)
    
    for category, data in info.items():
        print(f"\n📋 {category}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Save to file
    with open("diagnostic_report.txt", "w") as f:
        f.write("ASL2NL System Diagnostic Report\n")
        f.write("=" * 50 + "\n\n")
        
        for category, data in info.items():
            f.write(f"{category}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n✅ Report saved to diagnostic_report.txt")

if __name__ == "__main__":
    generate_report()
```

### Log File Analysis

Enable comprehensive logging:

```python
import logging
import datetime

# Configure logging
log_filename = f"asl2nl_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout application
logger.info("System started")
logger.debug(f"Camera index: {camera_index}")
logger.warning("Low confidence gesture detected")
logger.error("OpenAI API call failed")
```

### Remote Debugging

For complex issues, enable remote debugging:

```python
# Install debugpy: pip install debugpy
import debugpy

# Enable remote debugging
debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()

# Your code here
```

## 📞 Getting Help

### Before Asking for Help

1. **Run diagnostics:**
   ```bash
   python setup.py
   python -c "from docs.examples import diagnostic_example; diagnostic_example()"
   ```

2. **Check logs:**
   - Review error messages carefully
   - Note exact error text and line numbers
   - Check system logs for additional information

3. **Gather system information:**
   - Operating system and version
   - Python version
   - Package versions
   - Camera model and specifications

### Reporting Issues

When reporting issues, include:

1. **System Information:**
   - OS (Windows 10, macOS 12, Ubuntu 20.04, etc.)
   - Python version
   - Package versions (`pip list`)

2. **Error Details:**
   - Full error message and stack trace
   - Steps to reproduce the issue
   - Expected vs actual behavior

3. **Configuration:**
   - .env file contents (without API key)
   - Any custom modifications made
   - Camera setup and specifications

### Contact Information

- **GitHub Issues**: Report bugs and request features
- **Author**: Kenny Nguyen [@1kennect](https://github.com/1kennect)
- **Documentation**: Check docs/ folder for additional guides

### Community Resources

- **ASL Resources**: Learn proper ASL gestures and techniques
- **Computer Vision**: MediaPipe and OpenCV documentation
- **AI Integration**: OpenAI API documentation and best practices

---

This troubleshooting guide covers the most common issues encountered with the ASL2NL system. For issues not covered here, please refer to the diagnostic tools and contact information provided.
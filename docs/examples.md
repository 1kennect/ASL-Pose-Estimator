# Examples and Usage Guide

This guide provides practical examples and usage scenarios for the ASL2NL system.

## 📚 Table of Contents

- [Basic Usage Examples](#basic-usage-examples)
- [Advanced Usage Scenarios](#advanced-usage-scenarios)
- [Integration Examples](#integration-examples)
- [Customization Examples](#customization-examples)
- [Troubleshooting Examples](#troubleshooting-examples)

## 🚀 Basic Usage Examples

### Example 1: Simple Transcription Session

The most basic usage of the ASL2NL system:

```python
#!/usr/bin/env python3
"""
Basic ASL transcription example
"""
from asl_transcription_system import ASLTranscriptionSystem

def basic_transcription():
    """Run a basic transcription session."""
    # Initialize the system
    system = ASLTranscriptionSystem()
    
    # Start camera
    if not system.start_camera():
        print("❌ Failed to start camera")
        return
    
    # Run the system
    try:
        print("🚀 Starting ASL transcription...")
        print("   Press 't' to toggle transcription")
        print("   Press 'q' to quit")
        system.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    basic_transcription()
```

**Usage Steps:**
1. Run the script: `python basic_example.py`
2. Camera window opens with transcription OFF
3. Press `t` to toggle transcription ON
4. Perform ASL gestures: HELLO, THANK, YES, NO
5. Watch as words are recognized and sentences generated
6. Press `s` to save transcript
7. Press `q` to quit

### Example 2: Testing Individual Components

Test specific components before running the full system:

```python
#!/usr/bin/env python3
"""
Component testing example
"""
import cv2
import mediapipe as mp
from asl_gesture_recognition import recognize_asl_gesture

def test_components():
    """Test individual components."""
    print("🧪 Testing ASL2NL Components")
    
    # Test 1: Camera functionality
    print("\n1. Testing camera...")
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print("✅ Camera working")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame captured: {frame.shape}")
        cap.release()
    else:
        print("❌ Camera failed")
        return
    
    # Test 2: MediaPipe hands
    print("\n2. Testing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    print("✅ MediaPipe initialized")
    
    # Test 3: Gesture recognition
    print("\n3. Testing gesture recognition...")
    # Simulate hand landmarks for testing
    test_landmarks = [[0.5, 0.5, 0.0]] * 21  # 21 landmarks
    word, confidence = recognize_asl_gesture(test_landmarks)
    print(f"✅ Gesture recognition working: '{word}' ({confidence})")
    
    print("\n✅ All components working!")

if __name__ == "__main__":
    test_components()
```

### Example 3: Gesture Recognition Only

Use just the gesture recognition without AI processing:

```python
#!/usr/bin/env python3
"""
Gesture recognition only example
"""
import cv2
import mediapipe as mp
from asl_gesture_recognition import recognize_asl_gesture

def gesture_recognition_only():
    """Run gesture recognition without AI processing."""
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Start camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("👋 Gesture Recognition Mode")
    print("   Try: HELLO, THANK, YES, NO, PLEASE")
    print("   Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                
                # Recognize gesture
                word, confidence = recognize_asl_gesture(landmarks)
                
                if word and confidence > 0.6:
                    # Display recognized gesture
                    cv2.putText(frame, f"Gesture: {word} ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_recognition_only()
```

## 🎯 Advanced Usage Scenarios

### Example 4: Custom Configuration

Use custom settings and different OpenAI models:

```python
#!/usr/bin/env python3
"""
Advanced configuration example
"""
import os
from dotenv import load_dotenv
from asl_transcription_system import ASLTranscriptionSystem

def advanced_configuration():
    """Run with custom configuration."""
    # Load custom environment
    load_dotenv('.env.custom')  # Use custom env file
    
    # Initialize with custom settings
    system = ASLTranscriptionSystem(
        use_openai=True,
        model_name="gpt-4"  # Use GPT-4 for better results
    )
    
    # Custom camera configuration
    camera_index = int(os.getenv('CAMERA_INDEX', '0'))
    
    # Start with specific camera
    if not system.start_camera(camera_index):
        print(f"❌ Failed to start camera {camera_index}")
        # Try fallback cameras
        for idx in [0, 1, 2]:
            if system.start_camera(idx):
                print(f"✅ Using fallback camera {idx}")
                break
        else:
            print("❌ No working camera found")
            return
    
    # Custom buffer settings
    system.word_buffer.maxlen = 15  # Larger buffer
    system.min_words_for_ai = 2     # Process with fewer words
    system.ai_process_interval = 2.0  # Faster AI processing
    
    print("🚀 Advanced ASL System Running")
    print(f"   Model: {system.model_name}")
    print(f"   Buffer size: {system.word_buffer.maxlen}")
    print(f"   Min words for AI: {system.min_words_for_ai}")
    
    try:
        system.run()
    finally:
        system.cleanup()

if __name__ == "__main__":
    advanced_configuration()
```

### Example 5: Batch Processing

Process recorded video files instead of live camera:

```python
#!/usr/bin/env python3
"""
Batch video processing example
"""
import cv2
import mediapipe as mp
from asl_gesture_recognition import recognize_asl_gesture
from collections import deque
import openai
import os

class BatchASLProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize OpenAI
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Processing state
        self.word_buffer = deque(maxlen=10)
        self.sentences = []
    
    def process_video(self):
        """Process entire video file."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return False
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Processing video: {self.video_path}")
        print(f"   Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame for efficiency
            if frame_count % 10 == 0:
                self._process_frame(frame)
                
                # Progress indicator
                progress = (frame_count / total_frames) * 100
                print(f"\r⏳ Progress: {progress:.1f}%", end="")
        
        print(f"\n✅ Processed {frame_count} frames")
        
        # Process final buffer
        if len(self.word_buffer) > 0:
            self._process_buffer()
        
        # Save results
        self._save_results()
        
        cap.release()
        return True
    
    def _process_frame(self, frame):
        """Process single frame for gestures."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                word, confidence = recognize_asl_gesture(landmarks)
                
                if word and confidence > 0.7:  # Higher threshold for batch
                    self.word_buffer.append(word)
                    
                    # Process buffer when full
                    if len(self.word_buffer) >= 5:
                        self._process_buffer()
    
    def _process_buffer(self):
        """Process word buffer with AI."""
        if not self.word_buffer:
            return
        
        word_sequence = " ".join(list(self.word_buffer))
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Convert ASL word sequences to natural English sentences."},
                    {"role": "user", "content": f"ASL sequence: {word_sequence}"}
                ],
                max_tokens=50
            )
            
            sentence = response.choices[0].message.content.strip()
            if sentence and sentence not in ["INSUFFICIENT_CONTEXT", "RANDOM_LETTERS"]:
                self.sentences.append(sentence)
                print(f"\n📝 Generated: {sentence}")
        
        except Exception as e:
            print(f"\n⚠️  AI processing error: {e}")
        
        self.word_buffer.clear()
    
    def _save_results(self):
        """Save processing results."""
        with open(self.output_path, 'w') as f:
            f.write("ASL Video Processing Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Source: {self.video_path}\n\n")
            
            for i, sentence in enumerate(self.sentences, 1):
                f.write(f"{i}. {sentence}\n")
        
        print(f"✅ Results saved to: {self.output_path}")

def batch_processing_example():
    """Example of batch video processing."""
    video_path = "sample_asl_video.mp4"  # Your video file
    output_path = "asl_transcript_batch.txt"
    
    processor = BatchASLProcessor(video_path, output_path)
    
    if processor.process_video():
        print("🎉 Batch processing completed successfully!")
    else:
        print("❌ Batch processing failed")

if __name__ == "__main__":
    batch_processing_example()
```

## 🔧 Integration Examples

### Example 6: Flask Web API

Create a web API for ASL transcription:

```python
#!/usr/bin/env python3
"""
Flask web API example
"""
from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
from asl_gesture_recognition import recognize_asl_gesture
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # For single images
    max_num_hands=2,
    min_detection_confidence=0.7
)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize_gesture():
    """API endpoint for gesture recognition."""
    try:
        # Get image data
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gestures = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                word, confidence = recognize_asl_gesture(landmarks)
                
                if word and confidence > 0.6:
                    gestures.append({
                        'word': word,
                        'confidence': confidence
                    })
        
        return jsonify({
            'success': True,
            'gestures': gestures,
            'hands_detected': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ASL2NL API'})

if __name__ == '__main__':
    print("🌐 Starting ASL2NL Web API")
    print("   Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Example 7: Discord Bot Integration

Create a Discord bot that processes ASL images:

```python
#!/usr/bin/env python3
"""
Discord bot integration example
"""
import discord
from discord.ext import commands
import aiohttp
import base64
import cv2
import numpy as np
from asl_gesture_recognition import recognize_asl_gesture
import mediapipe as mp
import os

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!asl ', intents=intents)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

@bot.event
async def on_ready():
    print(f'🤖 ASL Bot ready! Logged in as {bot.user}')

@bot.command(name='recognize')
async def recognize_command(ctx):
    """Recognize ASL gestures from attached images."""
    if not ctx.message.attachments:
        await ctx.send("❌ Please attach an image with your ASL gesture!")
        return
    
    attachment = ctx.message.attachments[0]
    
    if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        await ctx.send("❌ Please attach a valid image file (PNG, JPG, JPEG)")
        return
    
    try:
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                image_data = await resp.read()
        
        # Process image
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            gestures = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                word, confidence = recognize_asl_gesture(landmarks)
                
                if word and confidence > 0.6:
                    gestures.append(f"**{word}** ({confidence:.2f})")
            
            if gestures:
                response = f"👋 Recognized gestures:\n" + "\n".join(gestures)
            else:
                response = "🤔 No clear gestures recognized. Try a clearer image!"
        else:
            response = "❌ No hands detected in the image."
        
        await ctx.send(response)
    
    except Exception as e:
        await ctx.send(f"❌ Error processing image: {str(e)}")

@bot.command(name='help')
async def help_command(ctx):
    """Show help information."""
    help_text = """
🤖 **ASL2NL Discord Bot Commands**

`!asl recognize` - Attach an image and I'll recognize ASL gestures
`!asl help` - Show this help message

**Supported Gestures:**
HELLO, THANK, YES, NO, PLEASE, SORRY, HELP, LOVE, GOOD, BAD, 
UNDERSTAND, NAME, WHAT, WHERE, WHO, and basic fingerspelling

**Tips:**
• Use clear, well-lit images
• Keep hands clearly visible
• Plain background works best
    """
    await ctx.send(help_text)

if __name__ == '__main__':
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("❌ Please set DISCORD_BOT_TOKEN environment variable")
    else:
        bot.run(token)
```

## 🎨 Customization Examples

### Example 8: Custom Gesture Recognition

Add your own custom gestures:

```python
#!/usr/bin/env python3
"""
Custom gesture recognition example
"""
from typing import List, Tuple
import numpy as np

def custom_recognize_asl_gesture(landmarks: List[List[float]]) -> Tuple[str, float]:
    """
    Custom gesture recognition with additional gestures.
    """
    if not landmarks or len(landmarks) < 21:
        return "", 0.0
    
    # Extract key points (same as original)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    palm_center = landmarks[9]
    wrist = landmarks[0]
    
    # Calculate finger states
    finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    finger_bases = [landmarks[3], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    finger_states = []
    
    for tip, base in zip(finger_tips, finger_bases):
        extended = tip[1] < base[1]
        angle = abs(tip[1] - base[1])
        finger_states.append({'extended': extended, 'angle': angle})
    
    # Extract states
    thumb_extended = finger_states[0]['extended']
    index_extended = finger_states[1]['extended']
    middle_extended = finger_states[2]['extended']
    ring_extended = finger_states[3]['extended']
    pinky_extended = finger_states[4]['extended']
    extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Calculate additional features
    palm_z = palm_center[2]
    palm_y = palm_center[1]
    palm_x = palm_center[0]
    
    # Custom gesture: PEACE (V-sign)
    if (index_extended and middle_extended and 
        not ring_extended and not pinky_extended and 
        abs(index_tip[0] - middle_tip[0]) > 0.05):
        return ("PEACE", 0.9)
    
    # Custom gesture: OK (thumb and index forming circle)
    if (thumb_extended and index_extended and 
        not middle_extended and not ring_extended and not pinky_extended and
        abs(thumb_tip[0] - index_tip[0]) < 0.03 and
        abs(thumb_tip[1] - index_tip[1]) < 0.03):
        return ("OK", 0.85)
    
    # Custom gesture: ROCK (closed fist with thumb out)
    if (thumb_extended and not index_extended and 
        not middle_extended and not ring_extended and not pinky_extended):
        return ("ROCK", 0.8)
    
    # Custom gesture: CALL_ME (thumb and pinky extended)
    if (thumb_extended and not index_extended and 
        not middle_extended and not ring_extended and pinky_extended):
        return ("CALL_ME", 0.8)
    
    # Fall back to original recognition
    from asl_gesture_recognition import recognize_asl_gesture
    return recognize_asl_gesture(landmarks)

# Integration example
def use_custom_recognition():
    """Example of using custom recognition."""
    from asl_transcription_system import ASLTranscriptionSystem
    
    # Create system
    system = ASLTranscriptionSystem()
    
    # Replace recognition function
    system._recognize_asl_gesture = lambda hand_landmarks: custom_recognize_asl_gesture(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    )
    
    print("🎨 Using custom gesture recognition")
    print("   Added gestures: PEACE, OK, ROCK, CALL_ME")
    
    # Run system
    if system.start_camera():
        system.run()

if __name__ == "__main__":
    use_custom_recognition()
```

### Example 9: Custom AI Processing

Use alternative AI providers or custom processing:

```python
#!/usr/bin/env python3
"""
Custom AI processing example
"""
import requests
import json
from asl_transcription_system import ASLTranscriptionSystem

class CustomAIProcessor:
    """Custom AI processor using different providers."""
    
    def __init__(self, provider="huggingface"):
        self.provider = provider
        
        if provider == "huggingface":
            self.api_url = "https://api-inference.huggingface.co/models/gpt2"
            self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        elif provider == "local":
            self.api_url = "http://localhost:11434/api/generate"  # Ollama
    
    def process_words(self, word_sequence: str) -> str:
        """Process word sequence with custom AI."""
        if self.provider == "huggingface":
            return self._process_huggingface(word_sequence)
        elif self.provider == "local":
            return self._process_local(word_sequence)
        elif self.provider == "rule_based":
            return self._process_rule_based(word_sequence)
        else:
            return word_sequence  # Fallback
    
    def _process_huggingface(self, word_sequence: str) -> str:
        """Use Hugging Face API."""
        try:
            prompt = f"Convert these ASL words to a sentence: {word_sequence}"
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": prompt, "parameters": {"max_length": 50}}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'].replace(prompt, '').strip()
            else:
                return word_sequence
        
        except Exception as e:
            print(f"⚠️  Hugging Face error: {e}")
            return word_sequence
    
    def _process_local(self, word_sequence: str) -> str:
        """Use local Ollama instance."""
        try:
            prompt = f"Convert ASL words to natural English: {word_sequence}"
            
            response = requests.post(
                self.api_url,
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return word_sequence
        
        except Exception as e:
            print(f"⚠️  Local AI error: {e}")
            return word_sequence
    
    def _process_rule_based(self, word_sequence: str) -> str:
        """Simple rule-based processing."""
        words = word_sequence.split()
        
        # Simple grammar rules
        if len(words) == 1:
            return words[0].lower()
        
        # Common patterns
        patterns = {
            ("HELLO", "HOW", "ARE", "YOU"): "Hello, how are you?",
            ("THANK", "YOU"): "Thank you.",
            ("YES", "PLEASE"): "Yes, please.",
            ("NO", "THANK", "YOU"): "No, thank you.",
            ("I", "LOVE", "YOU"): "I love you.",
            ("HELP", "PLEASE"): "Help, please.",
            ("GOOD", "MORNING"): "Good morning.",
        }
        
        # Check for exact matches
        word_tuple = tuple(words[:4])  # Check first 4 words
        for pattern, sentence in patterns.items():
            if word_tuple[:len(pattern)] == pattern:
                return sentence
        
        # Default: capitalize first word and add period
        return words[0].capitalize() + " " + " ".join(words[1:]).lower() + "."

def custom_ai_example():
    """Example using custom AI processing."""
    # Create system
    system = ASLTranscriptionSystem(use_openai=False)  # Disable OpenAI
    
    # Create custom AI processor
    ai_processor = CustomAIProcessor(provider="rule_based")
    
    # Replace AI processing function
    original_process = system._construct_sentence_with_ai
    system._construct_sentence_with_ai = ai_processor.process_words
    
    print("🧠 Using custom AI processing")
    print(f"   Provider: {ai_processor.provider}")
    
    # Run system
    if system.start_camera():
        system.run()

if __name__ == "__main__":
    custom_ai_example()
```

## 🛠 Troubleshooting Examples

### Example 10: Diagnostic Tool

Create a comprehensive diagnostic tool:

```python
#!/usr/bin/env python3
"""
Comprehensive diagnostic tool
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from asl_gesture_recognition import recognize_asl_gesture

class ASLDiagnostics:
    """Comprehensive diagnostic tool for ASL2NL system."""
    
    def __init__(self):
        self.results = {}
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("🔍 ASL2NL System Diagnostics")
        print("=" * 40)
        
        tests = [
            ("Python Environment", self.test_python),
            ("Dependencies", self.test_dependencies),
            ("Camera Hardware", self.test_cameras),
            ("MediaPipe", self.test_mediapipe),
            ("Gesture Recognition", self.test_gesture_recognition),
            ("OpenAI API", self.test_openai),
            ("Performance", self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                result = test_func()
                self.results[test_name] = result
                if result['status'] == 'pass':
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED - {result['message']}")
            except Exception as e:
                self.results[test_name] = {'status': 'error', 'message': str(e)}
                print(f"💥 {test_name}: ERROR - {e}")
        
        self.generate_report()
    
    def test_python(self):
        """Test Python environment."""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            return {'status': 'pass', 'version': f"{version.major}.{version.minor}.{version.micro}"}
        else:
            return {'status': 'fail', 'message': f"Python {version.major}.{version.minor} < 3.9"}
    
    def test_dependencies(self):
        """Test all required dependencies."""
        dependencies = [
            ('mediapipe', 'mediapipe'),
            ('opencv-python', 'cv2'),
            ('numpy', 'numpy'),
            ('openai', 'openai'),
            ('python-dotenv', 'dotenv')
        ]
        
        missing = []
        versions = {}
        
        for package, import_name in dependencies:
            try:
                module = __import__(import_name)
                if hasattr(module, '__version__'):
                    versions[package] = module.__version__
                else:
                    versions[package] = 'unknown'
            except ImportError:
                missing.append(package)
        
        if missing:
            return {'status': 'fail', 'message': f"Missing: {', '.join(missing)}"}
        else:
            return {'status': 'pass', 'versions': versions}
    
    def test_cameras(self):
        """Test camera availability and functionality."""
        working_cameras = []
        camera_info = {}
        
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    working_cameras.append(i)
                    camera_info[i] = {
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'frame_shape': frame.shape
                    }
                cap.release()
        
        if working_cameras:
            return {'status': 'pass', 'cameras': working_cameras, 'info': camera_info}
        else:
            return {'status': 'fail', 'message': 'No working cameras found'}
    
    def test_mediapipe(self):
        """Test MediaPipe functionality."""
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands()
            
            # Create test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = hands.process(test_image)
            
            return {'status': 'pass', 'version': mp.__version__}
        
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}
    
    def test_gesture_recognition(self):
        """Test gesture recognition module."""
        try:
            # Test with dummy landmarks
            test_landmarks = [[0.5, 0.5, 0.0]] * 21
            word, confidence = recognize_asl_gesture(test_landmarks)
            
            return {'status': 'pass', 'test_result': f"{word} ({confidence})"}
        
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}
    
    def test_openai(self):
        """Test OpenAI API connectivity."""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {'status': 'fail', 'message': 'OPENAI_API_KEY not set'}
            
            client = openai.OpenAI(api_key=api_key)
            
            # Test with minimal request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {'status': 'pass', 'response': response.choices[0].message.content}
        
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}
    
    def test_performance(self):
        """Test system performance."""
        try:
            # Test camera FPS
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                import time
                start_time = time.time()
                frame_count = 0
                
                for _ in range(30):  # Test 30 frames
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                
                cap.release()
                
                if fps >= 15:
                    return {'status': 'pass', 'fps': fps}
                else:
                    return {'status': 'fail', 'message': f'Low FPS: {fps:.1f}'}
            else:
                return {'status': 'fail', 'message': 'No camera available'}
        
        except Exception as e:
            return {'status': 'fail', 'message': str(e)}
    
    def generate_report(self):
        """Generate diagnostic report."""
        print("\n" + "=" * 40)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 40)
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'pass')
        total = len(self.results)
        
        print(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All systems operational!")
        else:
            print("⚠️  Some issues detected:")
            for test_name, result in self.results.items():
                if result['status'] != 'pass':
                    print(f"   • {test_name}: {result.get('message', 'Unknown error')}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if self.results.get('Camera Hardware', {}).get('status') != 'pass':
            print("   • Check camera connections and permissions")
        if self.results.get('OpenAI API', {}).get('status') != 'pass':
            print("   • Verify OPENAI_API_KEY in .env file")
        if self.results.get('Dependencies', {}).get('status') != 'pass':
            print("   • Run: pip install -r requirements.txt")

def diagnostic_example():
    """Run comprehensive diagnostics."""
    diagnostics = ASLDiagnostics()
    diagnostics.run_all_tests()

if __name__ == "__main__":
    diagnostic_example()
```

## 🎓 Learning Examples

### Example 11: Step-by-Step Tutorial

A guided tutorial for beginners:

```python
#!/usr/bin/env python3
"""
Interactive tutorial for ASL2NL
"""
import time
import cv2
from asl_transcription_system import ASLTranscriptionSystem

class ASLTutorial:
    """Interactive tutorial for learning ASL2NL."""
    
    def __init__(self):
        self.system = None
        self.current_step = 0
        self.steps = [
            self.step_introduction,
            self.step_camera_setup,
            self.step_basic_gestures,
            self.step_transcription,
            self.step_advanced_features,
            self.step_conclusion
        ]
    
    def run_tutorial(self):
        """Run the complete tutorial."""
        print("🎓 Welcome to the ASL2NL Tutorial!")
        print("   This will guide you through using the system")
        
        for i, step in enumerate(self.steps):
            self.current_step = i
            print(f"\n{'='*50}")
            print(f"Step {i+1}/{len(self.steps)}")
            print(f"{'='*50}")
            
            step()
            
            if i < len(self.steps) - 1:
                input("\nPress Enter to continue to the next step...")
    
    def step_introduction(self):
        """Introduction step."""
        print("📖 INTRODUCTION")
        print("ASL2NL converts American Sign Language gestures into English text.")
        print("\nKey features:")
        print("• Real-time gesture recognition")
        print("• AI-powered sentence construction") 
        print("• Support for common ASL signs and fingerspelling")
        
        print("\nSupported gestures:")
        gestures = ["HELLO", "THANK", "YES", "NO", "PLEASE", "SORRY", 
                   "HELP", "LOVE", "GOOD", "BAD", "UNDERSTAND"]
        print("• " + ", ".join(gestures))
    
    def step_camera_setup(self):
        """Camera setup step."""
        print("📹 CAMERA SETUP")
        print("Let's test your camera setup...")
        
        # Test camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera detected!")
            print("   A window will open showing your camera feed")
            print("   Make sure you can see your hands clearly")
            print("   Press 'q' to continue")
            
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, "Camera Test - Press 'q' to continue", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Camera Test', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera test completed!")
        else:
            print("❌ No camera detected. Please check your camera connection.")
    
    def step_basic_gestures(self):
        """Basic gestures step."""
        print("👋 BASIC GESTURES")
        print("Let's practice some basic ASL gestures!")
        
        gestures_to_try = [
            ("HELLO", "Open hand, palm facing forward"),
            ("THANK", "Flat hand moving from chin outward"),
            ("YES", "Closed fist nodding up and down"),
            ("NO", "Index finger moving side to side")
        ]
        
        print("\nWe'll test gesture recognition with these gestures:")
        for gesture, description in gestures_to_try:
            print(f"• {gesture}: {description}")
        
        print("\nStarting gesture recognition test...")
        print("Try each gesture when prompted!")
        
        # Initialize system
        self.system = ASLTranscriptionSystem()
        if self.system.start_camera():
            # Run gesture test
            self._run_gesture_practice(gestures_to_try)
            self.system.cleanup()
    
    def _run_gesture_practice(self, gestures_to_try):
        """Run gesture practice session."""
        import mediapipe as mp
        from asl_gesture_recognition import recognize_asl_gesture
        
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands()
        
        current_gesture_idx = 0
        gesture_recognized = False
        
        print(f"\nTry gesture: {gestures_to_try[current_gesture_idx][0]}")
        print(f"Description: {gestures_to_try[current_gesture_idx][1]}")
        
        while current_gesture_idx < len(gestures_to_try):
            ret, frame = self.system.cap.read()
            if not ret:
                continue
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Recognize gesture
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    word, confidence = recognize_asl_gesture(landmarks)
                    
                    target_gesture = gestures_to_try[current_gesture_idx][0]
                    
                    if word == target_gesture and confidence > 0.7:
                        gesture_recognized = True
                        cv2.putText(frame, f"✅ {word} RECOGNIZED!", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif word:
                        cv2.putText(frame, f"Detected: {word} ({confidence:.2f})", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, f"Try: {gestures_to_try[current_gesture_idx][0]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if gesture_recognized:
                cv2.putText(frame, "Press 'n' for next gesture or 'q' to continue", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Gesture Practice', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and gesture_recognized:
                current_gesture_idx += 1
                gesture_recognized = False
                if current_gesture_idx < len(gestures_to_try):
                    print(f"\nNext gesture: {gestures_to_try[current_gesture_idx][0]}")
                    print(f"Description: {gestures_to_try[current_gesture_idx][1]}")
        
        cv2.destroyAllWindows()
    
    def step_transcription(self):
        """Transcription step."""
        print("📝 TRANSCRIPTION")
        print("Now let's try the full transcription system!")
        print("\nThe system will:")
        print("1. Recognize your gestures")
        print("2. Buffer the words")
        print("3. Use AI to create sentences")
        
        print("\nImportant controls:")
        print("• Press 't' to toggle transcription ON/OFF")
        print("• Press 'f' to force process current buffer")
        print("• Press 'c' to clear buffer")
        print("• Press 's' to save transcript")
        print("• Press 'q' to quit")
        
        print("\nTry signing: HELLO THANK YOU")
        print("The system should generate: 'Hello, thank you.'")
        
        if not self.system:
            self.system = ASLTranscriptionSystem()
        
        if self.system.start_camera():
            print("\n🚀 Starting transcription system...")
            print("Remember to press 't' to turn transcription ON!")
            self.system.run()
    
    def step_advanced_features(self):
        """Advanced features step."""
        print("🚀 ADVANCED FEATURES")
        print("You've learned the basics! Here are advanced features:")
        
        print("\n1. Custom Configuration:")
        print("   • Edit .env file for API keys and settings")
        print("   • Adjust camera index if needed")
        
        print("\n2. Testing Tools:")
        print("   • python test_gestures.py - Test gesture recognition")
        print("   • python test_system.py - Test camera and MediaPipe")
        print("   • python camera_test.py - Test camera functionality")
        
        print("\n3. Customization:")
        print("   • Add custom gestures in asl_gesture_recognition.py")
        print("   • Modify AI prompts for different sentence styles")
        print("   • Adjust confidence thresholds")
        
        print("\n4. Integration:")
        print("   • Use as Python module in your projects")
        print("   • Create web APIs or Discord bots")
        print("   • Process recorded videos")
    
    def step_conclusion(self):
        """Conclusion step."""
        print("🎉 TUTORIAL COMPLETE!")
        print("Congratulations! You've learned how to use ASL2NL.")
        
        print("\n📚 What you learned:")
        print("• How to set up and test the camera")
        print("• Basic ASL gesture recognition")
        print("• Full transcription system usage")
        print("• Advanced features and customization")
        
        print("\n🔗 Next steps:")
        print("• Practice with more gestures")
        print("• Explore the documentation in docs/")
        print("• Try customizing the system")
        print("• Share your feedback and contributions!")
        
        print("\n📞 Need help?")
        print("• Check docs/troubleshooting.md")
        print("• Visit the GitHub repository")
        print("• Contact Kenny Nguyen (@1kennect)")
        
        print("\nThank you for using ASL2NL! 🙏")

def tutorial_example():
    """Run the interactive tutorial."""
    tutorial = ASLTutorial()
    tutorial.run_tutorial()

if __name__ == "__main__":
    tutorial_example()
```

---

These examples provide comprehensive coverage of the ASL2NL system usage, from basic operations to advanced customizations and integrations. Each example is designed to be practical and educational, helping users understand different aspects of the system.
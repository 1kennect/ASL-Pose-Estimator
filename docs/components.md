# Components Guide

This guide provides detailed information about the architecture and components of the ASL2NL system.

## 🏗 System Architecture

The ASL2NL system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    ASL2NL System                            │
├─────────────────────────────────────────────────────────────┤
│  User Interface Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   OpenCV GUI    │  │  Console Output │                 │
│  └─────────────────┘  └─────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         ASLTranscriptionSystem                          │ │
│  │  - Main orchestrator                                    │ │
│  │  - State management                                     │ │
│  │  - User interaction handling                            │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                           │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Gesture         │  │ AI Processing   │                 │
│  │ Recognition     │  │ (OpenAI)        │                 │
│  └─────────────────┘  └─────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   MediaPipe     │  │     OpenCV      │                 │
│  │ (Hand Detection)│  │ (Camera/Video)  │                 │
│  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Core Components

### 1. ASLTranscriptionSystem (`asl_transcription_system.py`)

The main orchestrator class that manages the entire transcription pipeline.

#### Responsibilities
- **Camera Management**: Initialize, configure, and manage video capture
- **Frame Processing**: Coordinate between computer vision and AI components
- **State Management**: Handle transcription state, buffers, and user interactions
- **UI Coordination**: Manage the graphical interface and user controls
- **Resource Management**: Handle cleanup and resource disposal

#### Key Subsystems

##### Camera Subsystem
```python
def start_camera(self, camera_index=1):
    """Camera initialization with fallback logic."""
    camera_indices = [camera_index, 0, 2, 1]
    
    for idx in camera_indices:
        self.cap = cv2.VideoCapture(idx)
        if self.cap.isOpened():
            # Test camera functionality
            return True
    return False
```

**Features:**
- Automatic camera detection and fallback
- Camera warming and initialization
- Frame rate monitoring
- Error recovery for camera failures

##### Word Buffer Management
```python
self.word_buffer = deque(maxlen=10)  # Circular buffer
self.word_timeout = 2.0              # Processing delay
self.ai_process_interval = 3.0       # AI call throttling
```

**Features:**
- Circular buffer for recent words
- Timeout-based processing
- AI call rate limiting
- Buffer state management

##### State Management
```python
self.transcription_active = False    # User control
self.current_sentence = ""          # Latest AI result
self.sentence_history = []          # All generated sentences
```

**Features:**
- User-controlled transcription toggle
- Sentence history tracking
- State persistence
- Clean state transitions

### 2. Gesture Recognition (`asl_gesture_recognition.py`)

Specialized module for converting hand landmarks into ASL gestures.

#### Architecture

```python
def recognize_asl_gesture(landmarks: List[List[float]]) -> Tuple[str, float]:
    """Main recognition pipeline."""
    # 1. Validate input
    if not landmarks or len(landmarks) < 21:
        return "", 0.0
    
    # 2. Extract key points
    key_points = extract_hand_features(landmarks)
    
    # 3. Analyze finger states
    finger_states = analyze_fingers(key_points)
    
    # 4. Classify gesture
    gesture, confidence = classify_asl_gesture(finger_states)
    
    return gesture, confidence
```

#### Feature Extraction

The system extracts several key features from hand landmarks:

##### Hand Landmarks (21 points)
```
    8   12  16  20
    |   |   |   |
4   |   |   |   |
|   |   |   |   |
0---5---9---13--17
    |   |   |   |
    6   10  14  18
    |   |   |   |
    7   11  15  19
```

##### Key Feature Points
- **Thumb**: Tip (4), IP (3), MCP (2), CMC (1)
- **Index**: Tip (8), DIP (7), PIP (6), MCP (5)
- **Middle**: Tip (12), DIP (11), PIP (10), MCP (9)
- **Ring**: Tip (16), DIP (15), PIP (14), MCP (13)
- **Pinky**: Tip (20), DIP (19), PIP (18), MCP (17)
- **Palm**: Center (9), Wrist (0)

##### Computed Features
```python
# Hand orientation
palm_normal = [palm_center[0] - wrist[0], 
               palm_center[1] - wrist[1], 
               palm_center[2] - wrist[2]]

# Finger extension analysis
for tip, base in zip(finger_tips, finger_bases):
    extended = tip[1] < base[1]  # Y-coordinate comparison
    angle = abs(tip[1] - base[1])  # Extension angle
    finger_states.append({'extended': extended, 'angle': angle})
```

#### Gesture Classification

The system uses rule-based classification with multiple criteria:

##### Primary ASL Signs
```python
# Example: HELLO gesture
if (extended_count >= 4 and palm_z > 0.5):
    return ("HELLO", 0.8)

# Example: YES gesture  
if (extended_count <= 1 and palm_y < 0.4):
    return ("YES", 0.8)
```

##### Fingerspelling Fallback
```python
# Letter mapping based on finger configuration
gesture_map = {
    (True, True, False, False, False): ("A", 0.7),
    (True, True, True, False, False): ("B", 0.7),
    # ... more mappings
}
```

### 3. Testing Components

#### Gesture Testing (`test_gestures.py`)

Interactive testing utility for gesture recognition.

**Features:**
- Real-time gesture visualization
- Performance metrics (FPS, recognition count)
- Interactive feedback
- Gesture instruction display

#### System Testing (`test_system.py`)

Basic system functionality testing.

**Features:**
- Camera functionality verification
- MediaPipe integration testing
- Hand detection validation
- Performance monitoring

#### Camera Testing (`camera_test.py`)

Comprehensive camera testing utility.

**Features:**
- Multi-camera detection
- Camera capability reporting
- Live preview testing
- Resolution and FPS analysis

### 4. Setup and Configuration (`setup.py`)

System setup and configuration management.

#### Dependency Management
```python
def check_dependencies():
    """Verify all required packages."""
    required_packages = [
        ('mediapipe', 'mediapipe'),
        ('opencv-python', 'cv2'), 
        ('numpy', 'numpy'),
        ('openai', 'openai'),
        ('python-dotenv', 'dotenv')
    ]
    # Validation logic...
```

#### Environment Configuration
```python
def create_env_file():
    """Create .env configuration file."""
    env_content = """
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-3.5-turbo
    CAMERA_INDEX=0
    """
    # File creation logic...
```

## 🔄 Data Flow

### 1. Frame Processing Pipeline

```
Camera → OpenCV → MediaPipe → Gesture Recognition → Word Buffer → AI Processing → Sentence Output
```

#### Detailed Flow

1. **Camera Capture**
   ```python
   ret, frame = self.cap.read()
   ```

2. **MediaPipe Processing**
   ```python
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = self.hands.process(rgb_frame)
   ```

3. **Hand Landmark Extraction**
   ```python
   landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
   ```

4. **Gesture Recognition**
   ```python
   word, confidence = recognize_asl_gesture(landmarks)
   ```

5. **Word Buffering**
   ```python
   if word and confidence > 0.6:
       self.word_buffer.append(word)
   ```

6. **AI Processing**
   ```python
   if len(self.word_buffer) >= self.min_words_for_ai:
       sentence = self._construct_sentence_with_ai(word_sequence)
   ```

### 2. State Management Flow

```
User Input → State Change → Buffer Update → Processing Trigger → Output Update
```

#### State Transitions

1. **Transcription Toggle** (`t` key)
   ```python
   self.transcription_active = not self.transcription_active
   if not self.transcription_active:
       self.word_buffer.clear()
   ```

2. **Buffer Management** (`c` key)
   ```python
   self.word_buffer.clear()
   self.current_sentence = ""
   ```

3. **Force Processing** (`f` key)
   ```python
   if len(self.word_buffer) > 0:
       self._process_word_buffer()
   ```

## ⚙️ Configuration System

### Environment Variables

The system uses a hierarchical configuration approach:

1. **Environment Variables** (`.env` file)
2. **Default Values** (in code)
3. **Runtime Parameters** (command line or API)

### Configuration Loading

```python
# Load environment variables
load_dotenv()

# Get configuration with defaults
api_key = os.getenv('OPENAI_API_KEY')
model_name = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
camera_index = int(os.getenv('CAMERA_INDEX', '1'))
```

### Runtime Configuration

```python
# MediaPipe configuration
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# System parameters
word_buffer = deque(maxlen=10)
word_timeout = 2.0
ai_process_interval = 3.0
```

## 🔌 Extension Points

### 1. Custom Gesture Recognition

```python
def custom_gesture_recognizer(landmarks):
    """Custom gesture recognition logic."""
    # Implement your custom logic
    if custom_condition(landmarks):
        return "CUSTOM_GESTURE", 0.9
    return "", 0.0

# Integration
system._recognize_asl_gesture = custom_gesture_recognizer
```

### 2. Alternative AI Providers

```python
class CustomAIProcessor:
    def __init__(self, provider="custom"):
        self.provider = provider
    
    def process_words(self, word_sequence):
        """Custom AI processing."""
        # Your AI logic here
        return processed_sentence

# Integration
system.ai_processor = CustomAIProcessor()
```

### 3. Custom UI Components

```python
def custom_ui_overlay(frame, system_state):
    """Add custom UI elements."""
    # Custom overlay logic
    cv2.putText(frame, f"Custom: {system_state.custom_data}", 
               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return frame

# Integration
system.ui_overlay_callback = custom_ui_overlay
```

## 🎯 Performance Considerations

### 1. Frame Processing Optimization

- **Frame Rate**: Target 30 FPS for smooth interaction
- **Processing Time**: Keep frame processing under 33ms
- **Memory Usage**: Efficient buffer management
- **CPU Usage**: Optimize MediaPipe settings

### 2. AI Processing Optimization

- **Rate Limiting**: Prevent excessive API calls
- **Caching**: Cache recent AI responses
- **Batching**: Process multiple words together
- **Fallback**: Graceful degradation without AI

### 3. Memory Management

- **Circular Buffers**: Fixed-size word and sentence buffers
- **Resource Cleanup**: Proper camera and MediaPipe cleanup
- **Garbage Collection**: Minimize object creation in loops

## 🔧 Debugging and Monitoring

### 1. Logging System

The system provides comprehensive logging:

```python
# Success indicators
print("✅ Component initialized")

# Warnings
print("⚠️  Potential issue detected")

# Errors
print("❌ Error occurred")

# Status updates
print("🔄 State changed")

# Information
print("📝 Processing complete")
```

### 2. Performance Metrics

- **FPS Monitoring**: Real-time frame rate display
- **Recognition Rate**: Gesture recognition success rate
- **AI Response Time**: OpenAI API call latency
- **Buffer Status**: Word buffer utilization

### 3. Debug Modes

Enable debug output for troubleshooting:

```python
# Enable debug logging
DEBUG = True

if DEBUG:
    print(f"Debug: Landmarks = {landmarks}")
    print(f"Debug: Gesture = {gesture}, Confidence = {confidence}")
```

---

This component architecture provides a solid foundation for the ASL2NL system while maintaining flexibility for future enhancements and customizations.
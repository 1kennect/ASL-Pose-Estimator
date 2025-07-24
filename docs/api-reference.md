# API Reference

This document provides comprehensive documentation for all public APIs, classes, and functions in the ASL2NL system.

## 📚 Table of Contents

- [ASLTranscriptionSystem](#asltranscriptionsystem)
- [Gesture Recognition Functions](#gesture-recognition-functions)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## ASLTranscriptionSystem

The main class that orchestrates the entire ASL transcription pipeline.

### Class Definition

```python
class ASLTranscriptionSystem:
    def __init__(self, use_openai=True, model_name="gpt-3.5-turbo")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_openai` | `bool` | `True` | Whether to use OpenAI for sentence construction |
| `model_name` | `str` | `"gpt-3.5-turbo"` | OpenAI model to use for transcription |

### Public Methods

#### `start_camera(camera_index=1)`

Initializes and starts the camera capture.

**Parameters:**
- `camera_index` (int): Camera index to use (default: 1)

**Returns:**
- `bool`: True if camera started successfully, False otherwise

**Example:**
```python
system = ASLTranscriptionSystem()
if system.start_camera(0):
    print("Camera started successfully")
else:
    print("Failed to start camera")
```

#### `process_frame(frame)`

Processes a single video frame for ASL recognition.

**Parameters:**
- `frame` (numpy.ndarray): BGR image frame from camera

**Returns:**
- `numpy.ndarray`: Processed frame with overlays and annotations

**Example:**
```python
ret, frame = cap.read()
if ret:
    processed_frame = system.process_frame(frame)
    cv2.imshow('ASL System', processed_frame)
```

#### `run()`

Starts the main processing loop with GUI interface.

**Parameters:** None

**Returns:** None

**Example:**
```python
system = ASLTranscriptionSystem()
system.start_camera()
system.run()  # Starts interactive session
```

#### `cleanup()`

Releases resources and cleans up the system.

**Parameters:** None

**Returns:** None

**Example:**
```python
try:
    system.run()
finally:
    system.cleanup()
```

### Properties

#### `transcription_active`

**Type:** `bool`

**Description:** Whether transcription is currently active

**Example:**
```python
if system.transcription_active:
    print("Transcription is running")
```

#### `current_sentence`

**Type:** `str`

**Description:** The current AI-generated sentence

**Example:**
```python
sentence = system.current_sentence
if sentence:
    print(f"Current sentence: {sentence}")
```

#### `word_buffer`

**Type:** `collections.deque`

**Description:** Buffer containing recently recognized words

**Example:**
```python
words = list(system.word_buffer)
print(f"Recent words: {' '.join(words)}")
```

#### `sentence_history`

**Type:** `list`

**Description:** History of all generated sentences

**Example:**
```python
for i, sentence in enumerate(system.sentence_history):
    print(f"{i+1}. {sentence}")
```

### Private Methods

These methods are internal but documented for developers who want to extend the system.

#### `_recognize_asl_gesture(hand_landmarks)`

Recognizes ASL gesture from MediaPipe hand landmarks.

**Parameters:**
- `hand_landmarks`: MediaPipe hand landmarks object

**Returns:**
- `tuple`: (recognized_word, confidence_score)

#### `_process_word_buffer()`

Processes the current word buffer using AI to generate sentences.

#### `_construct_sentence_with_ai(word_sequence)`

Uses OpenAI API to construct coherent sentences from word sequences.

**Parameters:**
- `word_sequence` (str): Space-separated sequence of recognized words

**Returns:**
- `str`: Generated sentence or empty string if unsuccessful

#### `_toggle_transcription()`

Toggles transcription on/off and manages buffer state.

#### `_clear_buffer()`

Clears the word buffer and current sentence.

#### `_force_process()`

Forces immediate processing of the current word buffer.

#### `_save_transcript()`

Saves the current transcript to a timestamped file.

## Gesture Recognition Functions

Functions in `asl_gesture_recognition.py` for recognizing ASL gestures.

### `recognize_asl_gesture(landmarks)`

Main function for recognizing ASL gestures from hand landmarks.

**Parameters:**
- `landmarks` (List[List[float]]): List of 21 hand landmark coordinates, each as [x, y, z]

**Returns:**
- `tuple`: (recognized_word, confidence_score)
  - `recognized_word` (str): The recognized ASL word or empty string
  - `confidence_score` (float): Confidence level between 0.0 and 1.0

**Example:**
```python
from asl_gesture_recognition import recognize_asl_gesture

# landmarks from MediaPipe
landmarks = [[x, y, z] for landmark in hand_landmarks.landmark 
             for x, y, z in [(landmark.x, landmark.y, landmark.z)]]

word, confidence = recognize_asl_gesture(landmarks)
if word and confidence > 0.6:
    print(f"Recognized: {word} (confidence: {confidence:.2f})")
```

### Supported Gestures

The system recognizes the following gestures with their typical confidence scores:

#### ASL Signs

| Gesture | Description | Confidence |
|---------|-------------|------------|
| `HELLO` | Open hand, palm forward | 0.8 |
| `THANK` | Flat hand from chin | 0.8 |
| `YES` | Closed fist nodding | 0.8 |
| `NO` | Index finger side to side | 0.8 |
| `PLEASE` | Flat hand rubbing | 0.8 |
| `SORRY` | Closed fist on chest | 0.8 |
| `HELP` | Open hand with thumb up | 0.8 |
| `LOVE` | Hands over heart | 0.8 |
| `GOOD` | Flat hand forward | 0.8 |
| `BAD` | Thumbs down | 0.8 |
| `UNDERSTAND` | Index finger to head | 0.8 |
| `NAME` | Two fingers pointing | 0.8 |
| `WHAT` | Open hand questioning | 0.8 |
| `WHERE` | Index finger pointing | 0.8 |
| `WHO` | Index finger to face | 0.8 |

#### Fingerspelling

| Letter | Hand Configuration | Confidence |
|--------|-------------------|------------|
| `A` | Closed fist, thumb up | 0.7 |
| `B` | Flat hand, fingers together | 0.7 |
| `C` | Curved hand | 0.7 |
| `D` | Index finger up, others folded | 0.7 |
| `E` | Fingers folded down | 0.7 |
| `I` | Pinky finger up | 0.7 |
| `L` | L-shape with thumb and index | 0.7 |
| `O` | Fingers forming circle | 0.7 |
| `Y` | Thumb and pinky extended | 0.7 |

## Utility Functions

### Setup Functions

#### `create_env_file()`

Creates a `.env` configuration file.

**Location:** `setup.py`

**Parameters:** None

**Returns:** None

**Example:**
```python
from setup import create_env_file
create_env_file()  # Creates .env with template
```

#### `check_dependencies()`

Verifies all required packages are installed.

**Location:** `setup.py`

**Parameters:** None

**Returns:**
- `bool`: True if all dependencies are satisfied

**Example:**
```python
from setup import check_dependencies
if check_dependencies():
    print("All dependencies installed")
```

### Testing Functions

#### `test_asl_gestures()`

Runs interactive gesture recognition test.

**Location:** `test_gestures.py`

**Parameters:** None

**Returns:**
- `bool`: True if test completed successfully

#### `test_camera_and_mediapipe()`

Tests camera functionality and MediaPipe integration.

**Location:** `test_system.py`

**Parameters:** None

**Returns:**
- `bool`: True if test passed

#### `test_cameras()`

Tests all available camera indices.

**Location:** `camera_test.py`

**Parameters:** None

**Returns:**
- `list`: List of working camera indices

## Configuration

### Environment Variables

The system uses environment variables for configuration via `.env` file:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | None | OpenAI API key (required) |
| `OPENAI_MODEL` | string | `gpt-3.5-turbo` | OpenAI model to use |
| `CAMERA_INDEX` | integer | 1 | Preferred camera index |

### System Parameters

#### Gesture Recognition Parameters

```python
# MediaPipe hand detection settings
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `static_image_mode` | False | Process video stream |
| `max_num_hands` | 2 | Maximum hands to detect |
| `min_detection_confidence` | 0.7 | Minimum confidence for detection |
| `min_tracking_confidence` | 0.5 | Minimum confidence for tracking |

#### Buffer and Timing Parameters

```python
# Word recognition settings
word_buffer = deque(maxlen=10)  # Store last 10 words
word_timeout = 2.0  # Seconds before processing
ai_process_interval = 3.0  # Minimum seconds between AI calls
min_words_for_ai = 3  # Minimum words before AI processing
```

## Error Handling

### Common Exceptions

#### `CameraError`

Raised when camera cannot be initialized or accessed.

```python
try:
    system.start_camera()
except Exception as e:
    print(f"Camera error: {e}")
```

#### `OpenAIError`

Raised when OpenAI API calls fail.

```python
try:
    sentence = system._construct_sentence_with_ai(words)
except Exception as e:
    print(f"AI processing error: {e}")
    # Fallback to word sequence
    sentence = " ".join(words)
```

#### `MediaPipeError`

Raised when MediaPipe processing fails.

```python
try:
    results = hands.process(frame)
except Exception as e:
    print(f"MediaPipe error: {e}")
    # Continue with next frame
```

### Error Recovery

The system implements several error recovery mechanisms:

1. **Camera Failures**: Automatic retry with different camera indices
2. **Frame Processing**: Skip failed frames and continue
3. **AI Processing**: Graceful fallback to raw word sequences
4. **Network Issues**: Retry OpenAI calls with exponential backoff

### Logging

The system provides console logging for debugging:

```python
print("✅ Success message")
print("⚠️  Warning message")  
print("❌ Error message")
print("🔄 Status change")
print("📝 Information")
```

## Integration Examples

### Custom Gesture Recognition

```python
def custom_recognize_gesture(landmarks):
    """Custom gesture recognition function."""
    # Your custom logic here
    if meets_custom_criteria(landmarks):
        return "CUSTOM_GESTURE", 0.9
    return "", 0.0

# Replace the recognition function
system._recognize_asl_gesture = custom_recognize_gesture
```

### Custom AI Processing

```python
def custom_ai_processor(word_sequence):
    """Custom AI processing function."""
    # Your custom logic here
    return f"Custom processing: {word_sequence}"

# Replace the AI processing function
system._construct_sentence_with_ai = custom_ai_processor
```

### Event Callbacks

```python
class CustomASLSystem(ASLTranscriptionSystem):
    def __init__(self):
        super().__init__()
        self.on_word_recognized = None
        self.on_sentence_generated = None
    
    def _process_word_buffer(self):
        """Override to add callbacks."""
        super()._process_word_buffer()
        if self.on_sentence_generated:
            self.on_sentence_generated(self.current_sentence)

# Usage
system = CustomASLSystem()
system.on_sentence_generated = lambda s: print(f"New sentence: {s}")
```

---

For more examples and advanced usage, see [examples.md](./examples.md).
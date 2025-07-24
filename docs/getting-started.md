# Getting Started with ASL2NL

This guide will help you set up and run the ASL2NL system on your machine.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.9 or higher
- **Camera**: USB webcam or built-in camera
- **Memory**: At least 4GB RAM (8GB recommended)
- **Internet**: Required for OpenAI API calls

### Hardware Requirements

- **Camera**: Any USB webcam or built-in camera with decent quality
- **Lighting**: Good lighting conditions for optimal hand detection
- **Background**: Plain background recommended for better recognition

## 🛠 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ASL2NL
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Setup Script

```bash
python setup.py
```

This script will:
- Check all dependencies are installed correctly
- Create a `.env` file for configuration
- Verify system compatibility

## ⚙️ Configuration

### OpenAI API Setup

1. **Get API Key**:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create an account or sign in
   - Generate a new API key

2. **Configure Environment**:
   - Open the `.env` file created by setup
   - Replace `your_openai_api_key_here` with your actual API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Optional Settings**:
   ```
   # Model selection (optional)
   OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4
   
   # Camera settings (optional)
   CAMERA_INDEX=0  # Change if default camera doesn't work
   ```

### Camera Configuration

The system will automatically detect and use available cameras. If you have multiple cameras:

1. **Test Available Cameras**:
   ```bash
   python camera_test.py
   ```

2. **Set Preferred Camera** (if needed):
   - Add `CAMERA_INDEX=X` to your `.env` file
   - Where X is the camera index that works best

## 🚀 First Run

### 1. Test System Components

Before running the main application, test individual components:

**Test Camera and MediaPipe**:
```bash
python test_system.py
```
- Verifies camera functionality
- Tests hand detection
- Shows FPS performance

**Test Gesture Recognition**:
```bash
python test_gestures.py
```
- Tests ASL gesture recognition
- Try different gestures to see recognition in action

### 2. Run Main Application

```bash
python asl_transcription_system.py
```

### 3. Using the Application

When the application starts:

1. **Camera Window Opens**: You'll see a live camera feed
2. **Transcription is OFF by default**: Press `t` to toggle it ON
3. **Perform ASL Gestures**: The system will recognize and buffer words
4. **View Results**: Sentences appear in the camera window and console

## 🎮 Controls

| Key | Action |
|-----|--------|
| `t` | Toggle transcription ON/OFF |
| `f` | Force process current word buffer |
| `c` | Clear word buffer |
| `s` | Save transcript to file |
| `q` | Quit application |

## 📊 Understanding the Interface

The camera window displays:

- **Current Sentence**: The AI-generated sentence from recent gestures
- **Word Buffer**: Recent recognized words waiting to be processed
- **Transcription Status**: Shows if transcription is ON or OFF
- **Hand Landmarks**: Visual overlay showing detected hand positions

## 🎯 Your First Gestures

Try these gestures to test the system:

1. **HELLO**: Open hand, palm facing forward
2. **THANK**: Flat hand moving from chin outward
3. **YES**: Closed fist nodding up and down
4. **NO**: Index finger moving side to side
5. **PLEASE**: Flat hand rubbing in circular motion on chest

## 🔧 Troubleshooting Quick Start

### Camera Issues
```bash
# Test all available cameras
python camera_test.py

# If camera doesn't work, try different index in .env:
CAMERA_INDEX=1  # or 2, 3, etc.
```

### Dependency Issues
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check installation
python setup.py
```

### OpenAI API Issues
- Verify API key is correct in `.env`
- Check your OpenAI account has credits
- Ensure internet connection is stable

### Performance Issues
- Ensure good lighting conditions
- Use plain background behind hands
- Close other applications using camera
- Try different camera if available

## 📁 Generated Files

The system creates several files during operation:

- **`.env`**: Configuration file (keep private)
- **`asl_transcript_YYYYMMDD_HHMMSS.txt`**: Saved transcripts
- **Log files**: Error logs and debug information

## 🔄 Next Steps

Once you have the system running:

1. **Explore Examples**: Check [examples.md](./examples.md) for usage scenarios
2. **Learn the API**: Read [api-reference.md](./api-reference.md) for customization
3. **Test Thoroughly**: Follow [testing.md](./testing.md) for comprehensive testing
4. **Customize**: See [components.md](./components.md) for modification guides

## 💡 Tips for Best Results

1. **Lighting**: Use consistent, bright lighting
2. **Background**: Plain, contrasting background works best
3. **Hand Position**: Keep hands clearly visible in camera frame
4. **Gesture Speed**: Perform gestures at moderate speed
5. **Practice**: Consistent gesture formation improves recognition

## 🆘 Need Help?

- **Common Issues**: Check [troubleshooting.md](./troubleshooting.md)
- **GitHub Issues**: Report bugs or ask questions
- **Contact**: Reach out to Kenny Nguyen [@1kennect](https://github.com/1kennect)

---

**Ready to start?** Run `python asl_transcription_system.py` and press `t` to begin transcription!
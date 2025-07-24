# ASL2NL Documentation

Welcome to the comprehensive documentation for **ASL2NL** - a real-time American Sign Language (ASL) gesture recognition and transcription system.

## 📖 Table of Contents

- [Getting Started](./getting-started.md) - Installation, setup, and first steps
- [API Reference](./api-reference.md) - Complete API documentation
- [Components Guide](./components.md) - Detailed component documentation
- [Examples](./examples.md) - Usage examples and tutorials
- [Testing Guide](./testing.md) - How to test the system
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
- [Contributing](./contributing.md) - How to contribute to the project

## 🎯 Project Overview

ASL2NL is a cutting-edge system that bridges the communication gap by providing real-time ASL gesture recognition and intelligent transcription to natural English sentences. Built with modern computer vision and AI technologies, it offers:

### ✨ Key Features

- **Real-time Recognition**: Live webcam-based ASL gesture detection
- **AI-Powered Transcription**: Converts gesture sequences into coherent English sentences
- **Modular Architecture**: Extensible design for easy customization
- **Multiple Testing Modes**: Comprehensive testing utilities
- **Cross-Platform**: Works on Windows, macOS, and Linux

### 🛠 Technology Stack

- **Computer Vision**: MediaPipe for hand landmark detection
- **Image Processing**: OpenCV for camera handling and frame processing  
- **AI Integration**: OpenAI GPT models for intelligent sentence construction
- **Language**: Python 3.9+
- **GUI**: OpenCV-based real-time display

### 📊 Supported Gestures

The system recognizes a comprehensive set of ASL gestures including:

**Common Signs**: HELLO, THANK, YES, NO, PLEASE, SORRY, HELP, LOVE, GOOD, BAD, UNDERSTAND, NAME, WHAT, WHERE, WHO

**Fingerspelling**: Basic alphabet letters (A, B, C, D, E, I, L, O, Y)

## 🚀 Quick Start

1. **Installation**
   ```bash
   git clone <repository-url>
   cd ASL2NL
   pip install -r requirements.txt
   ```

2. **Setup**
   ```bash
   python setup.py
   # Edit .env file with your OpenAI API key
   ```

3. **Run**
   ```bash
   python asl_transcription_system.py
   ```

For detailed instructions, see [Getting Started](./getting-started.md).

## 🏗 Project Structure

```
ASL2NL/
├── asl_transcription_system.py    # Main application
├── asl_gesture_recognition.py     # Core gesture recognition
├── test_gestures.py              # Gesture testing utility
├── test_system.py                # System testing utility  
├── camera_test.py                # Camera testing utility
├── setup.py                      # Setup and configuration
├── requirements.txt              # Dependencies
├── docs/                         # Documentation (this folder)
│   ├── README.md                 # This file
│   ├── getting-started.md        # Setup guide
│   ├── api-reference.md          # API documentation
│   ├── components.md             # Component details
│   ├── examples.md               # Usage examples
│   ├── testing.md                # Testing guide
│   ├── troubleshooting.md        # Common issues
│   └── contributing.md           # Contribution guide
└── .env                          # Environment configuration
```

## 📝 Author & Contact

**Created by**: Kenny Nguyen

**GitHub**: [@1kennect](https://github.com/1kennect)

For questions, issues, or contributions, please visit the GitHub repository or contact Kenny directly.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./contributing.md) for details on how to get started.

---

*This is a research and demonstration project. It is not intended for medical or legal communication purposes.*
# ASL2NL: ASL Gesture-to-Sentence Transcription

A real-time American Sign Language (ASL) gesture recognition and transcription system using MediaPipe, OpenCV, and OpenAI's API.

## Features
- Real-time webcam-based ASL gesture recognition (not just fingerspelling!)
- Converts recognized gestures into English sentences using an AI pipeline
- Modular, extensible, and ready for open source collaboration

## Project Structure
```
ASL2NL/
├── asl_transcription_system.py      # Main app: UI, camera, AI pipeline
├── asl_gesture_recognition.py       # All gesture recognition logic (imported by main/test)
├── test_gestures.py                 # Standalone gesture recognition test
├── test_system.py                   # Camera/MediaPipe test
├── requirements.txt                 # All dependencies
├── README.md                        # Usage, setup, and contribution guide
├── .gitignore                       # Ignore venv, .env, etc.
├── .env.example                     # Example env file (not secret)
├── LICENSE                          # MIT or similar
└── venv/                            # (gitignored) Local virtual environment
```

## Setup
1. **Clone the repo**
2. **Create a virtual environment**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure your OpenAI API key**
   - Copy `.env.example` to `.env` and add your key

## Usage
### Main System
```bash
python asl_transcription_system.py
```
- Press `t` to toggle transcription ON/OFF
- Press `f` to force process buffer
- Press `c` to clear buffer
- Press `s` to save transcript
- Press `q` to quit

### Gesture Test
```bash
python test_gestures.py
```
- Try gestures like HELLO, THANK, YES, NO, PLEASE, SORRY, HELP, LOVE, GOOD, BAD, etc.

### Camera/MediaPipe Test
```bash
python test_system.py
```

## Supported Gestures
- HELLO, THANK, YES, NO, PLEASE, SORRY, HELP, LOVE, GOOD, BAD, UNDERSTAND, NAME, WHAT, WHERE, WHO
- Fallback: basic fingerspelling (A, B, C, ...)

## Contributing
- Fork and PRs welcome!
- Add new gestures to `asl_gesture_recognition.py`
- See code comments for extension points

## License
MIT License. See LICENSE file.

## Authors
- Kenny (original author)

---
**Note:** This is a research/demonstration project and not a medical or legal communication tool. 

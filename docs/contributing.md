# Contributing to ASL2NL

Thank you for your interest in contributing to ASL2NL! This guide will help you get started with contributing to the project.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Community Guidelines](#community-guidelines)

## 🚀 Getting Started

### Project Overview

ASL2NL is an open-source project that provides real-time American Sign Language (ASL) gesture recognition and transcription. The project welcomes contributions in several areas:

- **Gesture Recognition**: Improving accuracy and adding new gestures
- **AI Integration**: Enhancing sentence construction and natural language processing
- **User Interface**: Improving usability and accessibility
- **Documentation**: Writing guides, tutorials, and API documentation
- **Testing**: Creating comprehensive test suites
- **Performance**: Optimizing speed and resource usage

### Ways to Contribute

1. **Bug Reports**: Report issues you encounter
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Submit bug fixes and new features
4. **Documentation**: Improve existing docs or write new ones
5. **Testing**: Add test cases and improve test coverage
6. **Community Support**: Help other users in discussions

### Before You Start

1. **Read the Documentation**: Familiarize yourself with the project by reading the docs
2. **Check Existing Issues**: Look for existing issues or feature requests
3. **Join Discussions**: Participate in project discussions and planning
4. **Understand the Codebase**: Review the code structure and architecture

## 🛠 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git for version control
- A webcam for testing
- OpenAI API key (for AI features)

### Fork and Clone

1. **Fork the Repository**:
   - Go to the GitHub repository
   - Click "Fork" to create your own copy

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ASL2NL.git
   cd ASL2NL
   ```

3. **Add Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/ASL2NL.git
   ```

### Development Environment

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Set Up Environment**:
   ```bash
   python setup.py
   # Edit .env file with your configuration
   ```

4. **Verify Setup**:
   ```bash
   python test_system.py
   python test_gestures.py
   ```

### Development Dependencies

Create `requirements-dev.txt` for development tools:

```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
unittest-xml-reporting>=3.2.0

# Code Quality
black>=22.0.0
flake8>=5.0.0
isort>=5.10.0
mypy>=0.991

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0

# Development Tools
pre-commit>=2.20.0
debugpy>=1.6.0
```

## 📋 Contributing Guidelines

### Issue Guidelines

#### Reporting Bugs

When reporting bugs, include:

1. **Clear Title**: Descriptive summary of the issue
2. **Environment Details**:
   - Operating system and version
   - Python version
   - Package versions (`pip list`)
   - Camera model/type

3. **Steps to Reproduce**:
   ```
   1. Run `python asl_transcription_system.py`
   2. Press 't' to enable transcription
   3. Perform HELLO gesture
   4. Observe that gesture is not recognized
   ```

4. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened

5. **Additional Information**:
   - Error messages and stack traces
   - Screenshots or videos if applicable
   - Configuration files (without sensitive data)

#### Feature Requests

When suggesting features:

1. **Use Case**: Explain why this feature would be useful
2. **Detailed Description**: Describe the feature in detail
3. **Implementation Ideas**: Suggest how it might be implemented
4. **Alternatives**: Mention alternative solutions you've considered

### Pull Request Guidelines

#### Before Creating a PR

1. **Check for Existing Work**: Ensure no one else is working on the same thing
2. **Create an Issue**: Discuss the change before implementing
3. **Follow Code Standards**: Ensure your code meets project standards
4. **Add Tests**: Include appropriate test coverage
5. **Update Documentation**: Update relevant documentation

#### PR Requirements

- **Clear Title**: Describe what the PR does
- **Detailed Description**: Explain the changes and motivation
- **Testing**: Show that changes work as expected
- **Documentation**: Update docs if needed
- **Small Scope**: Keep PRs focused on a single change

#### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## 📝 Code Standards

### Python Style Guide

Follow PEP 8 with these specific guidelines:

#### Code Formatting

```python
# Use Black for automatic formatting
black asl_transcription_system.py

# Line length: 88 characters (Black default)
# Use double quotes for strings
# Use f-strings for string formatting
```

#### Import Organization

```python
# Standard library imports
import os
import sys
import time
from collections import deque
from typing import List, Tuple, Optional

# Third-party imports
import cv2
import numpy as np
import mediapipe as mp
import openai
from dotenv import load_dotenv

# Local imports
from asl_gesture_recognition import recognize_asl_gesture
```

#### Naming Conventions

```python
# Classes: PascalCase
class ASLTranscriptionSystem:
    pass

# Functions and variables: snake_case
def recognize_gesture():
    gesture_confidence = 0.8
    return gesture_confidence

# Constants: UPPER_SNAKE_CASE
MAX_BUFFER_SIZE = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Private methods: _leading_underscore
def _process_internal_data(self):
    pass
```

#### Documentation Strings

```python
def recognize_asl_gesture(landmarks: List[List[float]]) -> Tuple[str, float]:
    """
    Recognize ASL gesture from hand landmarks.
    
    Args:
        landmarks: List of 21 hand landmark coordinates, each as [x, y, z]
        
    Returns:
        Tuple of (recognized_word, confidence_score)
        - recognized_word: The recognized ASL word or empty string
        - confidence_score: Confidence level between 0.0 and 1.0
        
    Example:
        >>> landmarks = [[0.5, 0.5, 0.0]] * 21
        >>> word, confidence = recognize_asl_gesture(landmarks)
        >>> print(f"Recognized: {word} ({confidence:.2f})")
    """
    pass
```

#### Error Handling

```python
# Use specific exception types
try:
    response = openai_client.chat.completions.create(...)
except openai.RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    time.sleep(1)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# Use context managers for resources
with open('transcript.txt', 'w') as f:
    f.write(transcript_content)
```

### Code Quality Tools

#### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

#### Type Hints

Use type hints for better code documentation:

```python
from typing import List, Tuple, Optional, Dict, Any

def process_gestures(
    gestures: List[str], 
    confidence_threshold: float = 0.6
) -> Optional[str]:
    """Process list of gestures and return sentence."""
    pass

class ASLTranscriptionSystem:
    def __init__(self, use_openai: bool = True, model_name: str = "gpt-3.5-turbo"):
        self.word_buffer: deque = deque(maxlen=10)
        self.sentence_history: List[str] = []
```

## 🧪 Testing Requirements

### Test Structure

Organize tests in the `tests/` directory:

```
tests/
├── __init__.py
├── unit/
│   ├── test_gesture_recognition.py
│   ├── test_ai_processing.py
│   └── test_system_components.py
├── integration/
│   ├── test_camera_integration.py
│   └── test_end_to_end.py
└── fixtures/
    ├── sample_landmarks.json
    └── test_data.py
```

### Writing Tests

#### Unit Tests

```python
import unittest
from unittest.mock import Mock, patch
import numpy as np

from asl_gesture_recognition import recognize_asl_gesture

class TestGestureRecognition(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.valid_landmarks = [[0.5, 0.5, 0.0]] * 21
        self.hello_landmarks = [
            # Specific landmark data for HELLO gesture
            [0.0, 0.0, 0.0],  # Wrist
            # ... more landmarks
        ]
    
    def test_recognize_hello_gesture(self):
        """Test recognition of HELLO gesture."""
        word, confidence = recognize_asl_gesture(self.hello_landmarks)
        
        self.assertEqual(word, "HELLO")
        self.assertGreater(confidence, 0.7)
    
    def test_invalid_landmarks_return_empty(self):
        """Test that invalid landmarks return empty result."""
        invalid_landmarks = [[0.5, 0.5, 0.0]] * 10  # Too few landmarks
        
        word, confidence = recognize_asl_gesture(invalid_landmarks)
        
        self.assertEqual(word, "")
        self.assertEqual(confidence, 0.0)
    
    @patch('asl_gesture_recognition.some_dependency')
    def test_gesture_recognition_with_mock(self, mock_dependency):
        """Test gesture recognition with mocked dependencies."""
        mock_dependency.return_value = True
        
        result = recognize_asl_gesture(self.valid_landmarks)
        
        mock_dependency.assert_called_once()
        self.assertIsNotNone(result)
```

#### Integration Tests

```python
import unittest
import cv2
import tempfile
import os

from asl_transcription_system import ASLTranscriptionSystem

class TestSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test system."""
        self.system = ASLTranscriptionSystem(use_openai=False)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        self.system.cleanup()
        # Clean up temp directory
    
    def test_camera_initialization(self):
        """Test camera can be initialized."""
        result = self.system.start_camera()
        
        # Should either succeed or fail gracefully
        self.assertIsInstance(result, bool)
        
        if result:
            self.assertIsNotNone(self.system.cap)
            self.assertTrue(self.system.cap.isOpened())
    
    def test_frame_processing_pipeline(self):
        """Test complete frame processing pipeline."""
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise exceptions
        processed_frame = self.system.process_frame(test_frame)
        
        self.assertEqual(processed_frame.shape, test_frame.shape)
```

### Test Coverage

Aim for high test coverage:

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Performance Tests

```python
import time
import unittest
from asl_transcription_system import ASLTranscriptionSystem

class TestPerformance(unittest.TestCase):
    
    def test_gesture_recognition_speed(self):
        """Test that gesture recognition is fast enough."""
        landmarks = [[0.5, 0.5, 0.0]] * 21
        
        start_time = time.time()
        for _ in range(100):
            recognize_asl_gesture(landmarks)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        self.assertLess(avg_time, 0.01)  # Less than 10ms per recognition
```

## 📤 Submitting Changes

### Git Workflow

#### Creating a Feature Branch

```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main

# Create feature branch
git checkout -b feature/add-new-gesture
```

#### Making Changes

```bash
# Make your changes
# Edit files, add tests, update docs

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add recognition for PEACE gesture

- Add PEACE gesture recognition logic
- Update gesture mapping with V-sign detection
- Add tests for PEACE gesture
- Update documentation with new gesture"
```

#### Commit Message Guidelines

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(gestures): add PEACE gesture recognition

fix(camera): resolve camera initialization on Windows

docs(api): update gesture recognition documentation

test(integration): add end-to-end system tests
```

#### Submitting Pull Request

```bash
# Push feature branch
git push origin feature/add-new-gesture

# Create pull request on GitHub
# Fill out PR template with details
```

### Review Process

#### What Reviewers Look For

1. **Code Quality**: Clean, readable, well-documented code
2. **Testing**: Adequate test coverage and passing tests
3. **Documentation**: Updated docs for new features
4. **Compatibility**: Works across supported platforms
5. **Performance**: No significant performance regressions

#### Addressing Review Comments

```bash
# Make requested changes
# Edit files based on feedback

# Commit changes
git add .
git commit -m "Address review comments

- Fix variable naming in gesture recognition
- Add missing docstrings
- Update test assertions"

# Push updates
git push origin feature/add-new-gesture
```

### Merging Guidelines

- **Squash and Merge**: For feature branches with multiple commits
- **Merge Commit**: For significant features that should preserve history
- **Rebase and Merge**: For simple, clean commits

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

1. **Be Respectful**: Treat everyone with respect and kindness
2. **Be Inclusive**: Welcome people of all backgrounds and experience levels
3. **Be Collaborative**: Work together constructively
4. **Be Patient**: Help others learn and grow
5. **Be Professional**: Maintain professional communication

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code review and collaboration

### Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **Documentation**: Author attribution
- **GitHub**: Contributor graphs and statistics

### Mentorship

New contributors can get help through:

- **Good First Issues**: Issues labeled for beginners
- **Mentorship**: Experienced contributors willing to help
- **Documentation**: Comprehensive guides and examples
- **Code Reviews**: Learning through feedback

## 🎯 Specific Contribution Areas

### Gesture Recognition

Areas for improvement:

1. **New Gestures**: Add support for more ASL signs
2. **Accuracy**: Improve recognition accuracy
3. **Robustness**: Handle varying lighting and hand positions
4. **Performance**: Optimize recognition speed

Example contribution:
```python
# Add new gesture to asl_gesture_recognition.py
def recognize_peace_gesture(finger_states, palm_position):
    """Recognize PEACE (V-sign) gesture."""
    index_extended = finger_states[1]['extended']
    middle_extended = finger_states[2]['extended']
    ring_extended = finger_states[3]['extended']
    pinky_extended = finger_states[4]['extended']
    
    if (index_extended and middle_extended and 
        not ring_extended and not pinky_extended):
        return "PEACE", 0.9
    
    return "", 0.0
```

### AI Integration

Areas for enhancement:

1. **Prompt Engineering**: Improve AI prompts for better results
2. **Alternative Providers**: Add support for other AI services
3. **Context Awareness**: Better understanding of conversation context
4. **Performance**: Faster AI processing

### User Interface

UI/UX improvements:

1. **Accessibility**: Better support for users with disabilities
2. **Customization**: User-configurable settings
3. **Visualization**: Better gesture feedback and status display
4. **Mobile Support**: Adapt for mobile devices

### Documentation

Documentation needs:

1. **Tutorials**: Step-by-step learning guides
2. **API Reference**: Complete API documentation
3. **Examples**: More usage examples and demos
4. **Translations**: Documentation in other languages

## 📊 Project Roadmap

### Short-term Goals (Next Release)

- [ ] Improve gesture recognition accuracy
- [ ] Add more ASL gestures
- [ ] Performance optimizations
- [ ] Better error handling
- [ ] Comprehensive test suite

### Medium-term Goals (Next 6 months)

- [ ] Mobile app support
- [ ] Real-time collaboration features
- [ ] Advanced AI integration
- [ ] Accessibility improvements
- [ ] Multi-language support

### Long-term Vision (Next Year)

- [ ] Production-ready deployment
- [ ] Educational platform integration
- [ ] Community-driven gesture database
- [ ] Research collaboration
- [ ] Commercial applications

## 📞 Getting Help

### For Contributors

- **Documentation**: Check existing docs first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Direct Contact**: Reach out to Kenny Nguyen [@1kennect](https://github.com/1kennect)

### Development Resources

- **Python**: [Python.org](https://python.org)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org)
- **MediaPipe**: [MediaPipe Documentation](https://mediapipe.dev)
- **OpenAI**: [OpenAI API Documentation](https://platform.openai.com/docs)

---

Thank you for contributing to ASL2NL! Your contributions help make ASL more accessible to everyone. 🙏

**Author**: Kenny Nguyen  
**GitHub**: [@1kennect](https://github.com/1kennect)  
**Project**: ASL2NL - Bridging communication through technology
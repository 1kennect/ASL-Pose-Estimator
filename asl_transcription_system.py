#!/usr/bin/env python3
"""
ASL Gesture Recognition + AI Pipeline System
Real-time ASL transcription using MediaPipe and AI for sentence construction
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import json
import os
from typing import List, Dict, Optional
import openai
from dotenv import load_dotenv
from asl_gesture_recognition import recognize_asl_gesture

# Load environment variables
load_dotenv()

class ASLTranscriptionSystem:
    def __init__(self, use_openai=True, model_name="gpt-3.5-turbo"):
        """Initialize the ASL transcription system."""
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Video capture
        self.cap = None
        
        # Word recognition and buffering
        self.word_buffer = deque(maxlen=10)  # Store last 10 recognized words
        self.current_word = ""
        self.word_confidence = 0.0
        self.last_word_time = time.time()
        self.word_timeout = 2.0  # Seconds to wait before processing word
        
        # Transcription controls
        self.transcription_active = False  # Start with transcription off
        self.last_hands_detected = False
        self.hands_detected_timeout = 3.0  # Seconds to wait after hands disappear
        self.last_hands_time = time.time()
        
        # AI processing controls
        self.last_ai_process_time = time.time()
        self.ai_process_interval = 3.0  # Minimum seconds between AI calls
        self.min_words_for_ai = 3  # Minimum words before calling AI
        
        # AI Pipeline setup
        self.use_openai = use_openai
        self.model_name = model_name
        self.openai_client = None
        self.sentence_history = []
        
        # Processing flags
        self.is_running = False
        self.current_sentence = ""
        
        # Initialize AI pipeline
        self._setup_ai_pipeline()
        
        print("✅ ASL Transcription System initialized")
        print(f"   AI Pipeline: {'OpenAI' if use_openai else 'Ollama'}")
        print(f"   Model: {model_name}")
        print("   Transcription starts OFF - Press 't' to toggle")
    
    def _setup_ai_pipeline(self):
        """Setup the AI pipeline (OpenAI or Ollama)."""
        if self.use_openai:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("⚠️  Warning: OPENAI_API_KEY not found in environment")
                print("   Please set OPENAI_API_KEY in your .env file")
                return
            self.openai_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized")
        else:
            # Ollama setup would go here
            print("✅ Ollama pipeline ready (not implemented yet)")
    
    def _recognize_asl_gesture(self, hand_landmarks) -> tuple:
        """
        Recognize ASL gesture from hand landmarks using external module.
        Returns: (recognized_word, confidence)
        """
        if not hand_landmarks:
            return "", 0.0
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return recognize_asl_gesture(landmarks)
    
    def _process_word_buffer(self):
        """Process the word buffer and construct sentences using AI."""
        if not self.word_buffer:
            return
        
        # Get recent words
        recent_words = list(self.word_buffer)
        word_sequence = " ".join(recent_words)
        
        # Use AI to construct sentence
        sentence = self._construct_sentence_with_ai(word_sequence)
        
        # Only update if we got a meaningful result
        if sentence and sentence != self.current_sentence:
            self.current_sentence = sentence
            self.sentence_history.append(sentence)
            print(f"📝 Sentence: {sentence}")
        elif not sentence:
            # Clear current sentence if AI returned empty (meaningless input)
            self.current_sentence = ""
    
    def _construct_sentence_with_ai(self, word_sequence: str) -> str:
        """Use AI to construct a coherent sentence from ASL word sequence."""
        if not self.openai_client:
            return word_sequence  # Fallback to raw sequence
        
        try:
            prompt = f"""
You are an expert ASL interpreter. Given these recognized ASL signs/words: "{word_sequence}"

Your task is to interpret this into a coherent English sentence. Follow these rules:

1. ONLY respond with a meaningful English sentence if the sequence makes sense
2. If the sequence is too short, unclear, or doesn't form a coherent thought, respond with "INSUFFICIENT_CONTEXT"
3. If the sequence is just random letters or doesn't form words, respond with "RANDOM_LETTERS"
4. Focus on natural ASL-to-English translation
5. Consider ASL grammar and context
6. If you can form a meaningful sentence, provide it without explanation
7. Keep responses concise and natural
8. Handle ASL words like HELLO, THANK, YES, NO, PLEASE, SORRY, HELP, LOVE, GOOD, BAD, etc.

Examples:
- "HELLO HOW ARE YOU" → "Hello, how are you?"
- "A B C D E" → "RANDOM_LETTERS"
- "THANK YOU" → "Thank you."
- "I LOVE YOU" → "I love you."
- "YES PLEASE" → "Yes, please."
- "NO THANK YOU" → "No, thank you."
- "HELP PLEASE" → "Help, please."
- "GOOD MORNING" → "Good morning."

Respond with only the interpreted sentence or one of the special responses (INSUFFICIENT_CONTEXT/RANDOM_LETTERS).
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert ASL interpreter. Only provide meaningful English sentences or special responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Only show meaningful results
            if result in ["INSUFFICIENT_CONTEXT", "RANDOM_LETTERS"]:
                return ""  # Don't show these responses
            else:
                return result
        
        except Exception as e:
            print(f"⚠️  AI processing error: {e}")
            return ""  # Don't show error messages
    
    def _toggle_transcription(self):
        """Toggle transcription on/off."""
        self.transcription_active = not self.transcription_active
        status = "ON" if self.transcription_active else "OFF"
        print(f"🔄 Transcription {status}")
        
        if not self.transcription_active:
            # Clear buffer when turning off
            self.word_buffer.clear()
            self.current_sentence = ""
            print("🧹 Word buffer cleared")
    
    def _clear_buffer(self):
        """Clear the word buffer."""
        self.word_buffer.clear()
        self.current_sentence = ""
        print("🧹 Word buffer cleared")
    
    def _force_process(self):
        """Force process the current word buffer."""
        if len(self.word_buffer) > 0:
            print("⚡ Force processing word buffer...")
            self._process_word_buffer()
        else:
            print("⚠️  No words in buffer to process")
    
    def start_camera(self, camera_index=1):  # Changed default from 0 to 1
        """Start the camera capture."""
        # Try multiple camera indices if the default fails
        camera_indices = [camera_index, 0, 2, 1]  # Try default (1), then 0, 2, then back to 1
        
        for idx in camera_indices:
            print(f"🔍 Trying camera index {idx}...")
            self.cap = cv2.VideoCapture(idx)
            
            if not self.cap.isOpened():
                print(f"❌ Could not open camera index {idx}")
                continue
            
            print(f"✅ Camera started (index: {idx})")
            
            # Give camera time to initialize
            print("⏳ Initializing camera...")
            time.sleep(2)  # Wait 2 seconds for camera to warm up
            
            # Test camera by reading a few frames
            success = False
            for i in range(5):
                ret, frame = self.cap.read()
                if not ret:
                    print(f"⚠️  Camera test frame {i+1} failed, retrying...")
                    time.sleep(0.5)
                else:
                    print(f"✅ Camera test frame {i+1} successful")
                    success = True
                    break
            
            if success:
                print("✅ Camera initialized successfully")
                return True
            else:
                print(f"❌ Camera index {idx} failed to initialize properly")
                self.cap.release()
        
        print("❌ Error: No working camera found")
        return False
    
    def process_frame(self, frame):
        """Process a single frame for ASL recognition."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Check if hands are detected
        hands_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        
        # Update hands detection timing
        if hands_detected:
            self.last_hands_time = time.time()
            self.last_hands_detected = True
        elif self.last_hands_detected:
            # Hands disappeared
            time_since_hands = time.time() - self.last_hands_time
            if time_since_hands > self.hands_detected_timeout:
                self.last_hands_detected = False
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Only recognize gestures if transcription is active
                if self.transcription_active:
                    # Recognize ASL gesture
                    word, confidence = self._recognize_asl_gesture(hand_landmarks)
                    
                    if word and confidence > 0.6:
                        current_time = time.time()
                        
                        # Check if enough time has passed since last word
                        if current_time - self.last_word_time > 0.5:  # 500ms debounce
                            self.word_buffer.append(word)
                            self.last_word_time = current_time
                            print(f"👋 Recognized: {word} (confidence: {confidence:.2f})")
        
        # Only process word buffer if transcription is active and hands were recently detected
        if (self.transcription_active and 
            self.last_hands_detected and 
            time.time() - self.last_word_time > self.word_timeout and 
            len(self.word_buffer) >= self.min_words_for_ai and
            time.time() - self.last_ai_process_time > self.ai_process_interval):
            self._process_word_buffer()
            self.last_ai_process_time = time.time()
        
        return frame
    
    def run(self):
        """Main processing loop."""
        if not self.cap:
            print("❌ Error: Camera not started")
            return
        
        self.is_running = True
        print("🚀 Starting ASL transcription...")
        print("   Press 'q' to quit")
        print("   Press 's' to save transcript")
        print("   Press 't' to toggle transcription ON/OFF")
        print("   Press 'c' to clear word buffer")
        print("   Press 'f' to force process current buffer")
        print("   Transcription starts OFF - Press 't' to begin")
        
        # Additional camera warm-up time
        print("⏳ Warming up camera for live processing...")
        time.sleep(1)
        
        consecutive_failures = 0
        max_failures = 10  # Allow more failures before giving up
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"⚠️  Could not read frame (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("❌ Error: Too many consecutive frame failures")
                    break
                
                time.sleep(0.1)  # Brief pause before retry
                continue
            
            # Reset failure counter on successful frame
            consecutive_failures = 0
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add UI elements
            self._add_ui_elements(processed_frame)
            
            # Display frame
            cv2.imshow('ASL Transcription System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_transcript()
            elif key == ord('t'):
                self._toggle_transcription()
            elif key == ord('c'):
                self._clear_buffer()
            elif key == ord('f'):
                self._force_process()
        
        self.cleanup()
    
    def _add_ui_elements(self, frame):
        """Add UI elements to the frame."""
        # Add current sentence
        cv2.putText(frame, f"Current: {self.current_sentence}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add word buffer
        buffer_text = f"Buffer: {' '.join(list(self.word_buffer))}"
        cv2.putText(frame, buffer_text, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add transcription status
        status_text = "Transcription: OFF"
        if self.transcription_active:
            status_text = "Transcription: ON"
        cv2.putText(frame, status_text, 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to toggle, 'c' to clear, 'f' to force process", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_transcript(self):
        """Save the current transcript to a file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"asl_transcript_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("ASL Transcription Transcript\n")
            f.write("=" * 30 + "\n\n")
            for i, sentence in enumerate(self.sentence_history, 1):
                f.write(f"{i}. {sentence}\n")
        
        print(f"✅ Transcript saved to: {filename}")
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ System cleaned up")

def main():
    """Main function to run the ASL transcription system."""
    print("🎯 ASL Gesture Recognition + AI Pipeline System")
    print("=" * 50)
    
    # Initialize system
    system = ASLTranscriptionSystem(use_openai=True)
    
    # Start camera
    if not system.start_camera():
        return
    
    # Run the system
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main() 
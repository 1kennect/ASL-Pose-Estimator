#!/usr/bin/env python3
"""
Test script for ASL Gesture Recognition
Shows the new gesture recognition system in action
"""

import cv2
import mediapipe as mp
import time
from asl_gesture_recognition import recognize_asl_gesture

def test_asl_gestures():
    """Test the new ASL gesture recognition system."""
    print("🎯 Testing ASL Gesture Recognition System")
    print("=" * 50)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize hands detector
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Test camera
    cap = cv2.VideoCapture(1)  # Use camera index 1
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    print("✅ MediaPipe initialized successfully")
    print("📹 Starting gesture recognition test...")
    print("   Try these ASL signs:")
    print("   - HELLO (open hand, palm forward)")
    print("   - THANK YOU (flat hand from chin)")
    print("   - YES (fist moving up/down)")
    print("   - NO (index finger side to side)")
    print("   - PLEASE (flat hand rubbing)")
    print("   - HELP (flat hand with thumb up)")
    print("   - LOVE (hands over heart)")
    print("   - GOOD (flat hand forward)")
    print("   - BAD (thumbs down)")
    print("   Press 'q' to quit")
    
    start_time = time.time()
    frame_count = 0
    gesture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Test gesture recognition
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                # Analyze gesture
                gesture, confidence = recognize_asl_gesture(landmarks)
                
                if gesture:
                    gesture_count += 1
                    cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add info text
        cv2.putText(frame, "ASL Gesture Recognition Test", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            cv2.putText(frame, f"Hands detected: {len(results.multi_hand_landmarks)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands detected", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow('ASL Gesture Recognition Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Test completed successfully!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Gestures detected: {gesture_count}")
    print(f"   Average FPS: {fps:.1f}")
    
    return True

if __name__ == "__main__":
    test_asl_gestures() 
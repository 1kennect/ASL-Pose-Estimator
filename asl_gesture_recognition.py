"""
asl_gesture_recognition.py
All gesture recognition logic for ASL2NL project.
"""

from typing import List, Tuple

def recognize_asl_gesture(landmarks: List[List[float]]) -> Tuple[str, float]:
    """
    Recognize ASL gesture from hand landmarks.
    Returns: (recognized_word, confidence)
    """
    if not landmarks or len(landmarks) < 21:
        return "", 0.0
    
    # Extract key hand points
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    palm_center = landmarks[9]
    wrist = landmarks[0]
    
    # Calculate hand orientation and position
    palm_normal = [palm_center[0] - wrist[0], palm_center[1] - wrist[1], palm_center[2] - wrist[2]]
    hand_position = {'x': palm_center[0], 'y': palm_center[1], 'z': palm_center[2]}
    
    # Analyze finger states
    finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    finger_bases = [landmarks[3], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    finger_states = []
    for tip, base in zip(finger_tips, finger_bases):
        extended = tip[1] < base[1]
        angle = abs(tip[1] - base[1])
        finger_states.append({'extended': extended, 'angle': angle, 'position': tip})
    
    # Extract finger states
    thumb_extended = finger_states[0]['extended']
    index_extended = finger_states[1]['extended']
    middle_extended = finger_states[2]['extended']
    ring_extended = finger_states[3]['extended']
    pinky_extended = finger_states[4]['extended']
    extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    palm_z = hand_position['z']
    palm_y = hand_position['y']
    
    # ASL Sign Classification (expand as needed)
    if (extended_count >= 4 and palm_z > 0.5):
        return ("HELLO", 0.8)
    if (index_extended and not middle_extended and not ring_extended and palm_z < 0.3):
        return ("THANK", 0.8)
    if (extended_count <= 1 and palm_y < 0.4):
        return ("YES", 0.8)
    if (index_extended and not middle_extended and palm_z < 0.5):
        return ("NO", 0.8)
    if (extended_count >= 4 and palm_z < 0.4):
        return ("PLEASE", 0.8)
    if (extended_count <= 1 and palm_z < 0.3):
        return ("SORRY", 0.8)
    if (extended_count >= 4 and thumb_extended and palm_z > 0.4):
        return ("HELP", 0.8)
    if (extended_count >= 3 and palm_y < 0.3):
        return ("LOVE", 0.8)
    if (extended_count >= 4 and palm_z < 0.4):
        return ("GOOD", 0.8)
    if (thumb_extended and not index_extended and palm_z < 0.4):
        return ("BAD", 0.8)
    if (index_extended and not middle_extended and palm_z > 0.6):
        return ("UNDERSTAND", 0.8)
    if (index_extended and middle_extended and not ring_extended and not pinky_extended):
        return ("NAME", 0.8)
    if (extended_count >= 4 and palm_z > 0.5):
        return ("WHAT", 0.8)
    if (index_extended and not middle_extended and palm_z < 0.3):
        return ("WHERE", 0.8)
    if (index_extended and not middle_extended and palm_z > 0.4):
        return ("WHO", 0.8)
    # Fallback: basic character recognition
    extended = [f['extended'] for f in finger_states]
    gesture_map = {
        (True, True, False, False, False): ("A", 0.7),
        (True, True, True, False, False): ("B", 0.7),
        (True, True, True, True, True): ("C", 0.7),
        (False, True, True, True, True): ("D", 0.7),
        (True, False, False, False, False): ("E", 0.7),
        (False, True, False, False, False): ("I", 0.7),
        (False, False, True, False, False): ("L", 0.7),
        (False, False, False, True, False): ("O", 0.7),
        (False, False, False, False, True): ("Y", 0.7),
    }
    gesture_key = tuple(extended)
    if gesture_key in gesture_map:
        return gesture_map[gesture_key]
    return "", 0.0 
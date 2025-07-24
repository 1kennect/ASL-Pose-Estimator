#!/usr/bin/env python3
"""
Simple camera test script
Tests camera functionality and lists available cameras
"""

import cv2
import time

def test_cameras():
    """Test all available camera indices."""
    print("🔍 Testing available cameras...")
    
    working_cameras = []
    
    for i in range(5):  # Test indices 0-4
        print(f"\n📹 Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if not cap.isOpened():
            print(f"❌ Camera {i}: Not available")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Camera {i}: Cannot read frames")
            cap.release()
            continue
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera {i}: Working")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        working_cameras.append(i)
        cap.release()
    
    return working_cameras

def test_specific_camera(camera_index):
    """Test a specific camera with live preview."""
    print(f"🎥 Testing camera {camera_index} with live preview...")
    print("Press 'q' to quit, any other key to continue")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return False
    
    # Give camera time to initialize
    time.sleep(2)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame")
            break
        
        frame_count += 1
        
        # Add info to frame
        cv2.putText(frame, f"Camera {camera_index} - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_index} Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Camera {camera_index} test completed")
    print(f"   Frames processed: {frame_count}")
    print(f"   Average FPS: {fps:.1f}")
    
    return True

def main():
    """Main test function."""
    print("🎯 Camera Test Utility")
    print("=" * 30)
    
    # Test all cameras
    working_cameras = test_cameras()
    
    if not working_cameras:
        print("\n❌ No working cameras found!")
        return
    
    print(f"\n✅ Found {len(working_cameras)} working camera(s): {working_cameras}")
    
    # Ask user if they want to test a specific camera
    if len(working_cameras) > 1:
        print("\nWhich camera would you like to test with live preview?")
        for cam in working_cameras:
            print(f"  {cam}: Camera {cam}")
        
        try:
            choice = int(input("Enter camera index (or press Enter to skip): "))
            if choice in working_cameras:
                test_specific_camera(choice)
        except ValueError:
            print("Skipping live preview test")
    else:
        # Test the only available camera
        print(f"\nTesting camera {working_cameras[0]} with live preview...")
        test_specific_camera(working_cameras[0])
    
    print("\n✅ Camera test completed!")

if __name__ == "__main__":
    main() 
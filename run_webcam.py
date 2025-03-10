import cv2
import mediapipe as mp
import numpy as np
from thresholds import get_thresholds_beginner  # or get_thresholds_pro if desired
from process_frame import ProcessFrame
from utils import get_mediapipe_pose

def main():
    # Choose the mode; here we use Beginner mode
    thresholds = get_thresholds_beginner()
    # If you want Pro mode, you can import and use get_thresholds_pro() instead.
    
    # Create an instance of your ProcessFrame class
    # Set flip_frame as needed (True flips the frame horizontally)
    process_frame_instance = ProcessFrame(thresholds=thresholds, flip_frame=False)
    
    # Initialize MediaPipe Pose from your utils
    pose = get_mediapipe_pose()

    # Open the webcam. Using cv2.CAP_DSHOW on Windows can sometimes help.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        # Convert the frame from BGR (OpenCV default) to RGB,
        # because MediaPipe expects RGB images.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame using your ProcessFrame instance.
        # The process() method returns the processed frame and an optional play_sound flag.
        processed_frame_rgb, _ = process_frame_instance.process(frame_rgb, pose)
        
        # Convert the processed frame back to BGR for display with OpenCV.
        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        cv2.imshow("Squat Analysis", processed_frame_bgr)
        
        # Press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

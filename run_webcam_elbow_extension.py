# run_webcam_elbow_extension.py
import cv2
import mediapipe as mp
from thresholds_elbow_extension import get_thresholds_elbow_extension
from process_frame_elbow_extension import ProcessFrameElbowExtension

def main():
    thresholds = get_thresholds_elbow_extension()
    processor = ProcessFrameElbowExtension(thresholds, flip_frame=False)
    pose = mp.solutions.pose.Pose()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, _ = processor.process(frame_rgb, pose)
        # Convert processed frame back to BGR.
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Elbow Extension Analysis", processed_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

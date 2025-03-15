import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from thresholds import get_thresholds_beginner
from process_frame import ProcessFrame
from utils import get_mediapipe_pose

def main():
    thresholds = get_thresholds_beginner()
    process_frame_instance = ProcessFrame(thresholds=thresholds, flip_frame=False)
    pose = get_mediapipe_pose()
    try:
        model = tf.keras.models.load_model("model_squats.h5")
        model.summary()
    except Exception as e:
        model = None
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame_rgb, _ = process_frame_instance.process(frame_rgb, pose)
        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Squat Analysis", processed_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

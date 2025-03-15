import cv2
import mediapipe as mp
from thresholds_leg_extension import get_thresholds_leg_extension
from process_frame_leg_extension import ProcessFrameLegExtension
import tensorflow as tf

def main():
    thresholds = get_thresholds_leg_extension()
    processor = ProcessFrameLegExtension(thresholds, flip_frame=False)
    pose = mp.solutions.pose.Pose()
    try:
        model = tf.keras.models.load_model("model_leg_extension.h5")
        model.summary()
    except Exception as e:
        model = None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, _ = processor.process(frame_rgb, pose)
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Leg Extension Analysis", processed_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

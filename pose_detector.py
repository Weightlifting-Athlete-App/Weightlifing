import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import time

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Real-time pose detection using a trained exercise posture model with debugging."
)
parser.add_argument(
    "--exercise",
    type=str,
    default="shoulder_elbow_flexion",
    help=("Name of the exercise model to use. Options: shoulder_elbow_flexion, pendulum, crossover_arm_stretch.")
)
args = parser.parse_args()
exercise = args.exercise.lower()
model_filename = f"{exercise}_model.pkl"

print("OpenCV version:", cv2.__version__)
print("cv2 module file:", cv2.__file__)

print(f"Loading the trained model from '{model_filename}' ...")
model = joblib.load(model_filename)
print("Model loaded successfully.")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def extract_joint_angles(landmarks):
    keypoints = {
        'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
        'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
    }
    angles = {
        'left_elbow_angle': calculate_angle(keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist']),
        'right_elbow_angle': calculate_angle(keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist']),
        'left_knee_angle': calculate_angle(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle']),
        'right_knee_angle': calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle']),
        'left_shoulder_angle': calculate_angle(keypoints['left_hip'], keypoints['left_shoulder'], keypoints['left_elbow']),
        'right_shoulder_angle': calculate_angle(keypoints['right_hip'], keypoints['right_shoulder'], keypoints['right_elbow']),
    }
    return angles

# Open the webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam opened successfully.")

# Create and position the display window
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow('Pose Detection', 100, 100)  # Force the window to appear at position (100, 100)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured. Exiting loop.")
        break

    frame_count += 1
    print(f"Frame {frame_count} captured.")

    # Convert frame to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    try:
        results = pose.process(image)
    except Exception as e:
        print("Error in pose.process:", e)
        continue

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            angles = extract_joint_angles(landmarks)
            print("Extracted angles:", angles)

            angle_data = np.array([
                angles['left_elbow_angle'],
                angles['right_elbow_angle'],
                angles['left_knee_angle'],
                angles['right_knee_angle'],
                angles['left_shoulder_angle'],
                angles['right_shoulder_angle']
            ]).reshape(1, -1)

            prediction = model.predict(angle_data)
            label_text = "Correct Posture" if prediction[0] == 1 else "Incorrect Posture"
            cv2.putText(image, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if prediction[0] == 1 else (0, 0, 255), 2)
        except Exception as e:
            print("Error during pose processing:", e)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        print("No pose landmarks detected in this frame.")

    cv2.imshow('Pose Detection', image)

    key = cv2.waitKey(1) & 0xFF  # Using a shorter delay (1ms)
    if key == ord('q'):
        print("Exiting loop because 'q' was pressed.")
        break

elapsed = time.time() - start_time
print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} fps).")

cap.release()
cv2.destroyAllWindows()
print("Released webcam and closed all windows.")

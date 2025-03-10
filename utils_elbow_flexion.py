import cv2
import mediapipe as mp
import numpy as np

def draw_text(img, msg, pos, font_scale=1, text_color=(0, 255, 0), thickness=2, text_color_bg=None):
    """
    Draw text on the image. If a background color is provided, draw a rectangle behind the text.
    """
    if text_color_bg is not None:
        (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = pos
        # Draw a filled rectangle slightly larger than the text.
        cv2.rectangle(img, (x-2, y - text_h - 2), (x + text_w + 2, y + 2), text_color_bg, -1)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def find_angle(a, b, c):
    """
    Calculate the angle at point b given points a, b, and c.
    Typically: a = shoulder, b = elbow, c = wrist.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_landmark_array(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def get_landmark_features(landmarks, dict_features, feature, frame_width, frame_height):
    """
    Extract landmark coordinates for a specific feature.
    For example, if feature == 'right', returns:
       (shoulder, elbow, wrist)
    using the indexes specified in dict_features.
    """
    if feature in dict_features:
        feat = dict_features[feature]
        shoulder = get_landmark_array(landmarks[feat['shoulder']], frame_width, frame_height)
        elbow    = get_landmark_array(landmarks[feat['elbow']], frame_width, frame_height)
        wrist    = get_landmark_array(landmarks[feat['wrist']], frame_width, frame_height)
        return shoulder, elbow, wrist
    else:
        return None

def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                      ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                 )
    return pose
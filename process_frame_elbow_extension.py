import cv2
import mediapipe as mp
import numpy as np
from utils_elbow_extension import draw_text, find_angle, get_landmark_array

class ProcessFrameElbowExtension:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.stage = None
        self.rep_incorrect = False
        self.max_ext_angle = None
        self.last_time = 0

    @property
    def state_tracker(self):
        return {
            "CORRECT_REPS": self.correct_reps,
            "INCORRECT_REPS": self.incorrect_reps
        }

    def _get_state(self, angle):
        flexed_range = self.thresholds['ELBOW_ANGLE']['FLEXED']
        extended_range = self.thresholds['ELBOW_ANGLE']['EXTENDED']
        if flexed_range[0] <= angle <= flexed_range[1]:
            return 'flexed'
        elif extended_range[0] <= angle <= extended_range[1]:
            return 'extended'
        else:
            return None

    def process(self, frame, pose):
        play_sound = None
        h, w, _ = frame.shape
        keypoints = pose.process(frame)
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            shoulder = get_landmark_array(ps_lm.landmark[12], w, h)
            elbow = get_landmark_array(ps_lm.landmark[14], w, h)
            wrist = get_landmark_array(ps_lm.landmark[16], w, h)
            cv2.circle(frame, shoulder, 7, (0, 255, 255), -1)
            cv2.circle(frame, elbow, 7, (0, 255, 255), -1)
            cv2.circle(frame, wrist, 7, (0, 255, 255), -1)
            angle = find_angle(shoulder, elbow, wrist)
            draw_text(frame, f'Angle: {int(angle)}', pos=(50, 50), text_color=(255,255,0), font_scale=1)
            current_state = self._get_state(angle)
            print(f"Angle: {angle}, Current State: {current_state}, Prev State: {self.stage}")
            incorrect_flag = False
            if abs(elbow[0] - shoulder[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                incorrect_flag = True
            flexed_range = self.thresholds['ELBOW_ANGLE']['FLEXED']
            extended_range = self.thresholds['ELBOW_ANGLE']['EXTENDED']
            if current_state == 'extended':
                if self.max_ext_angle is None or angle > self.max_ext_angle:
                    self.max_ext_angle = angle
            if self.stage == 'extended' and current_state == 'flexed':
                if self.max_ext_angle is None or self.max_ext_angle < extended_range[0] or incorrect_flag:
                    self.incorrect_reps += 1
                else:
                    self.correct_reps += 1
                self.max_ext_angle = None
            if current_state is not None:
                self.stage = current_state
            draw_text(frame, f'Correct Reps: {self.correct_reps}', pos=(50, 100), font_scale=1.2, text_color=(0,255,0))
            draw_text(frame, f'Incorrect Reps: {self.incorrect_reps}', pos=(50, 140), font_scale=1.2, text_color=(0,0,255))
            correction = ""
            if angle < extended_range[0]:
                correction += "EXTEND YOUR ARM! "
            if abs(elbow[0] - shoulder[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                correction += "KEEP ELBOW STABLE! "
            if correction:
                draw_text(frame, correction, pos=(50, 180), font_scale=1, text_color=(0,0,255))
        else:
            self.max_ext_angle = None
        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        return frame, play_sound

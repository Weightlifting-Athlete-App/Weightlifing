import cv2
import time
import numpy as np
from utils_leg_extension import draw_text, find_angle, get_landmark_array

class ProcessFrameLegExtension:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame
        self.correct_reps = 0
        self.incorrect_reps = 0
        # For leg extension, assume "bent" is the starting position and "extended" is the fully straight leg.
        self.stage = None  # Expected to be 'bent' or 'extended'
        self.rep_incorrect = False
        self.max_ext_angle = None  # Track maximum knee angle during extension phase.
        self.last_time = time.time()
    
    def _get_state(self, angle):
        """
        Determine the current state based on the knee angle.
        'bent' if the knee angle is within the bent range,
        'extended' if it is within the extended range.
        """
        bent_range = self.thresholds['KNEE_ANGLE']['BENT']
        extended_range = self.thresholds['KNEE_ANGLE']['EXTENDED']
        if bent_range[0] <= angle <= bent_range[1]:
            return 'bent'
        elif extended_range[0] <= angle <= extended_range[1]:
            return 'extended'
        else:
            return None
    
    def process(self, frame, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape

        keypoints = pose.process(frame)
        if keypoints.pose_landmarks:
            landmarks = keypoints.pose_landmarks.landmark
            # Right leg landmarks: hip (24), knee (26), ankle (28)
            hip   = get_landmark_array(landmarks[24], frame_width, frame_height)
            knee  = get_landmark_array(landmarks[26], frame_width, frame_height)
            ankle = get_landmark_array(landmarks[28], frame_width, frame_height)
            
            # Draw landmark markers.
            cv2.circle(frame, hip, 5, (0, 255, 255), -1)
            cv2.circle(frame, knee, 5, (0, 255, 255), -1)
            cv2.circle(frame, ankle, 5, (0, 255, 255), -1)
            
            # Compute the knee angle.
            angle = find_angle(hip, knee, ankle)
            draw_text(frame, f'Knee Angle: {int(angle)}', (50, 50), font_scale=1, text_color=(255,255,0))
            
            bent_range = self.thresholds['KNEE_ANGLE']['BENT']
            extended_range = self.thresholds['KNEE_ANGLE']['EXTENDED']
            
            # For alignment checking (optional)
            alignment_threshold = self.thresholds.get('KNEE_ALIGNMENT_THRESHOLD', None)
            incorrect_flag = False
            if alignment_threshold is not None:
                # For example, you might check horizontal distance between hip and knee.
                if abs(knee[0] - hip[0]) > alignment_threshold:
                    incorrect_flag = True
            
            current_state = self._get_state(angle)
            # Debug print
            print(f"Angle: {angle}, Current State: {current_state}, Prev State: {self.stage}")
            
            # While in the extended state, update maximum extension angle.
            if current_state == 'extended':
                if self.max_ext_angle is None or angle > self.max_ext_angle:
                    self.max_ext_angle = angle
            
            # Rep counting logic:
            # Instead of counting rep immediately on reaching extended state,
            # we count a rep when transitioning from extended back to bent.
            if self.stage == 'extended' and current_state == 'bent':
                # If we didn't reach a high enough extension or alignment is off, mark as incorrect.
                if (self.max_ext_angle is None or self.max_ext_angle < extended_range[0]) or incorrect_flag:
                    self.incorrect_reps += 1
                else:
                    self.correct_reps += 1
                # Reset max extension for next rep.
                self.max_ext_angle = None
            # Update stage if current state is valid.
            if current_state is not None:
                self.stage = current_state
            
            # Display rep counts.
            draw_text(frame, f'Correct Reps: {self.correct_reps}', (50, 100), font_scale=1.2, text_color=(0,255,0))
            draw_text(frame, f'Incorrect Reps: {self.incorrect_reps}', (50, 140), font_scale=1.2, text_color=(0,0,255))
            
            # Corrective feedback.
            correction = ""
            if angle < extended_range[0]:
                correction += "Straighten your leg! "
            if alignment_threshold is not None and abs(knee[0] - hip[0]) > alignment_threshold:
                correction += "Keep knee stable! "
            if correction:
                draw_text(frame, correction, (50, 180), font_scale=1, text_color=(0,0,255))
        else:
            self.max_ext_angle = None  # Reset if no landmarks detected.
        
        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        
        return frame, play_sound

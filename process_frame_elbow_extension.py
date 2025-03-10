import cv2
import time
import numpy as np
from utils_elbow_extension import draw_text, find_angle, get_landmark_array

class ProcessFrameElbowExtension:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame
        self.correct_reps = 0
        self.incorrect_reps = 0
        # For extension, we consider 'down' as the flexed state and 'up' as the extended state.
        self.stage = None  # expected to be 'down' (flexed) or 'up' (extended)
        self.rep_incorrect = False
        self.max_ext_angle = None  # Track maximum angle achieved during the extension phase.
        self.last_time = time.time()
    
    def _get_state(self, angle):
        """
        Determine the current state based on the elbow angle.
        For elbow extension:
          - 'flexed' when the arm is bent (angle within FLEXED range)
          - 'extended' when the arm is straight (angle within EXTENDED range)
        """
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
        frame_height, frame_width, _ = frame.shape

        keypoints = pose.process(frame)
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            # Get right-arm landmarks: shoulder (12), elbow (14), wrist (16)
            shoulder = get_landmark_array(ps_lm.landmark[12], frame_width, frame_height)
            elbow    = get_landmark_array(ps_lm.landmark[14], frame_width, frame_height)
            wrist    = get_landmark_array(ps_lm.landmark[16], frame_width, frame_height)

            # Draw joint markers.
            cv2.circle(frame, shoulder, 7, (0, 255, 255), -1)
            cv2.circle(frame, elbow, 7, (0, 255, 255), -1)
            cv2.circle(frame, wrist, 7, (0, 255, 255), -1)

            # Compute the elbow angle.
            angle = find_angle(shoulder, elbow, wrist)
            draw_text(frame, f'Angle: {int(angle)}', pos=(50, 50), text_color=(255,255,0), font_scale=1)

            # Determine current state.
            current_state = self._get_state(angle)
            # For debugging:
            print(f"Angle: {angle}, Current State: {current_state}, Prev State: {self.stage}")

            # Evaluate corrective condition: horizontal alignment.
            incorrect_flag = False
            if abs(elbow[0] - shoulder[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                incorrect_flag = True

            flexed_range = self.thresholds['ELBOW_ANGLE']['FLEXED']
            extended_range = self.thresholds['ELBOW_ANGLE']['EXTENDED']

            # While in the extended phase, track the maximum angle reached.
            if current_state == 'extended':
                if self.max_ext_angle is None or angle > self.max_ext_angle:
                    self.max_ext_angle = angle

            # Rep counting logic:
            # Instead of counting immediately on transition from flexed to extended,
            # we count the rep when the state transitions from extended back to flexed.
            # This gives time for the extension phase to complete.
            if self.stage == 'extended' and current_state == 'flexed':
                # If we never reached a high enough extension or if there's misalignment, mark as incorrect.
                if self.max_ext_angle is None or self.max_ext_angle < extended_range[0] or incorrect_flag:
                    self.incorrect_reps += 1
                else:
                    self.correct_reps += 1
                # Reset for next rep.
                self.max_ext_angle = None
            # Update stage:
            if current_state is not None:
                self.stage = current_state

            # Display rep counts.
            draw_text(frame, f'Correct Reps: {self.correct_reps}', pos=(50, 100), font_scale=1.2, text_color=(0,255,0))
            draw_text(frame, f'Incorrect Reps: {self.incorrect_reps}', pos=(50, 140), font_scale=1.2, text_color=(0,0,255))

            # Corrective feedback.
            correction = ""
            if angle < extended_range[0]:
                correction += "EXTEND YOUR ARM! "
            if abs(elbow[0] - shoulder[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                correction += "KEEP ELBOW STABLE! "
            if correction:
                draw_text(frame, correction, pos=(50, 180), font_scale=1, text_color=(0,0,255))
        else:
            # Reset the max extension angle if no landmarks are detected.
            self.max_ext_angle = None

        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        return frame, play_sound

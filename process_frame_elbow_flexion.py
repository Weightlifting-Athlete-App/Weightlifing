import time
import cv2
import numpy as np
from utils_elbow_flexion import find_angle, get_landmark_features, draw_text  # Using your common utils

class ProcessFrameElbowFlexion:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        
        # Font and line settings.
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        
        # Colors in BGR.
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255)
        }
        
        # Right-arm landmarks (using same format as squats).
        self.dict_features = {
            'right': {
                'shoulder': 12,
                'elbow': 14,
                'wrist': 16
            }
        }
        
        # State tracker similar to squats.
        self.state_tracker = {
            'state_seq': [],
            'CORRECT_REPS': 0,
            'INCORRECT_REPS': 0,
            'prev_state': None,
            'curr_state': None
        }
        
        # For tracking the minimum angle reached during the flexed phase.
        self.min_flex_angle = None
        
        # Feedback messages mapping (if needed).
        self.FEEDBACK_ID_MAP = {
            0: ('KEEP ELBOW CLOSE', 100, (0, 0, 255)),
            1: ('CURL MORE', 140, (0, 0, 255))
        }
    
    def _get_state(self, angle):
        """
        Determine the current state based on the elbow angle.
        For elbow flexion:
          - 'flexed' when the arm is curled (angle within FLEXED range)
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
    
    def process(self, frame: np.array, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape

        # Process the image using MediaPipe.
        keypoints = pose.process(frame)
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            # Get right-arm landmarks: shoulder, elbow, wrist.
            right_shldr_coord, right_elbow_coord, right_wrist_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            # Compute the elbow angle (shoulder-elbow-wrist).
            angle = find_angle(right_shldr_coord, right_elbow_coord, right_wrist_coord)
            draw_text(frame, f'Angle: {int(angle)}', pos=(50, 50),
                      text_color=(255, 255, 0), font_scale=1)

            # Draw joint markers.
            cv2.circle(frame, right_shldr_coord, 7, self.COLORS['yellow'], -1)
            cv2.circle(frame, right_elbow_coord, 7, self.COLORS['yellow'], -1)
            cv2.circle(frame, right_wrist_coord, 7, self.COLORS['yellow'], -1)

            # Determine the current state.
            current_state = self._get_state(angle)
            self.state_tracker['curr_state'] = current_state
            print(f"Angle: {angle}, Current State: {current_state}, Prev State: {self.state_tracker['prev_state']}")

            # Evaluate corrective conditions.
            incorrect_flag = False
            if abs(right_elbow_coord[0] - right_shldr_coord[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                incorrect_flag = True

            flexed_range = self.thresholds['ELBOW_ANGLE']['FLEXED']
            extended_range = self.thresholds['ELBOW_ANGLE']['EXTENDED']

            # While in the flexed phase, update the minimum angle.
            if current_state == 'flexed':
                if self.min_flex_angle is None or angle < self.min_flex_angle:
                    self.min_flex_angle = angle
            # If not in flexed state, but still in the range between flexed and extended,
            # we can also update the min (optional, depending on how your motion behaves)
            elif current_state is None and angle > flexed_range[1] and angle < extended_range[0]:
                if self.min_flex_angle is None or angle < self.min_flex_angle:
                    self.min_flex_angle = angle

            # Rep counting logic:
            # When transitioning from flexed to extended, count a rep.
            if current_state == 'extended' and self.state_tracker['prev_state'] == 'flexed':
                # If the minimum angle during the flexed phase never reached the proper low threshold
                # (i.e., is greater than the upper bound of the flexed range) OR the horizontal misalignment exists,
                # mark the rep as incorrect.
                if incorrect_flag or (self.min_flex_angle is not None and self.min_flex_angle > flexed_range[1]):
                    self.state_tracker['INCORRECT_REPS'] += 1
                else:
                    self.state_tracker['CORRECT_REPS'] += 1
                self.state_tracker['state_seq'] = []  # reset sequence after counting rep
                self.min_flex_angle = None  # reset for next rep
            else:
                self.state_tracker['state_seq'].append(current_state)

            self.state_tracker['prev_state'] = current_state

            # Display the rep counts.
            draw_text(frame, "CORRECT: " + str(self.state_tracker['CORRECT_REPS']),
                      pos=(int(frame_width * 0.68), 30), text_color=(255, 255, 230),
                      font_scale=0.7)
            draw_text(frame, "INCORRECT: " + str(self.state_tracker['INCORRECT_REPS']),
                      pos=(int(frame_width * 0.68), 80), text_color=(255, 255, 230),
                      font_scale=0.7)

            # Corrective feedback.
            feedback = ""
            if abs(right_elbow_coord[0] - right_shldr_coord[0]) > self.thresholds.get('ELBOW_ALIGNMENT_THRESHOLD', 50):
                feedback += "KEEP ELBOW CLOSE! "
            if angle > flexed_range[1] and angle < extended_range[0]:
                feedback += "CURL MORE! "
            if feedback:
                draw_text(frame, feedback, pos=(50, 100), text_color=(0, 0, 255), font_scale=0.7)
        else:
            # When no landmarks are detected, optionally reset min_flex_angle.
            self.min_flex_angle = None

        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        return frame, play_sound

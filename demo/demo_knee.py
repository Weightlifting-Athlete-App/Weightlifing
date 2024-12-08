import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import _init_paths  # Ensure this is correctly set up
import models  # Ensure your models are correctly imported
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

# Define COCO keypoint indexes and colors
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Skeleton connections for visualization
SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6],
    [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

# Colors for visualization
CocoColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

NUM_KPTS = 17
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at point p2 formed by p1 and p3.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # If either vector length is zero, return a default angle to avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp the value to avoid numerical errors leading to NaN
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_pose(keypoints, img, offset_x=-100, offset_y=100):
    """
    Draw the keypoints and the skeleton lines on the image.
    """
    assert keypoints.shape == (NUM_KPTS, 2), f"Expected keypoints shape (17,2), got {keypoints.shape}"
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a]
        x_b, y_b = keypoints[kpt_b]
        x_a_shifted = int(x_a) + offset_x
        y_a_shifted = int(y_a) + offset_y
        x_b_shifted = int(x_b) + offset_x
        y_b_shifted = int(y_b) + offset_y
        cv2.line(img, (x_a_shifted, y_a_shifted), (x_b_shifted, y_b_shifted), CocoColors[i], 2)
    for i, (x, y) in enumerate(keypoints):
        x_shifted = int(x) + offset_x
        y_shifted = int(y) + offset_y
        cv2.circle(img, (x_shifted, y_shifted), 6, CocoColors[i], -1)
        cv2.putText(img, f"{COCO_KEYPOINT_INDEXES[i]}", (x_shifted, y_shifted-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CocoColors[i], 1, cv2.LINE_AA)

def box_to_center_scale(box, model_image_width, model_image_height):
    """
    Convert a box to center, scale information required for pose transformation.
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    center[0] = bottom_left_corner[0] + box_width * 0.5
    center[1] = bottom_left_corner[1] + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

class SquatDetector:
    """
    Detect and evaluate squat exercises based on knee angles.
    """
    def __init__(self, left_threshold_down=90, left_threshold_up=160,
                 right_threshold_down=90, right_threshold_up=160):
        self.left_threshold_down = left_threshold_down
        self.left_threshold_up = left_threshold_up
        self.right_threshold_down = right_threshold_down
        self.right_threshold_up = right_threshold_up

        self.state = 'standing'  # Possible states: 'standing', 'squatting'
        self.count = 0

    def update(self, left_angle, right_angle):
        """
        Update the state based on current knee angles and return feedback messages.
        """
        feedback = []

        # Check if both knees are below the down threshold to transition to 'squatting'
        if self.state == 'standing':
            if left_angle < self.left_threshold_down and right_angle < self.right_threshold_down:
                self.state = 'squatting'
                feedback.append("Squat Down")
        elif self.state == 'squatting':
            # Check if both knees are above the up threshold to transition back to 'standing'
            if left_angle > self.left_threshold_up and right_angle > self.right_threshold_up:
                self.state = 'standing'
                self.count += 1
                feedback.append("Stand Up")
        return feedback

def get_person_detection_boxes(model, img, threshold=0.5):
    """
    Detect persons in the image using the object detection model.
    Returns a list of bounding boxes for detected persons.
    """
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    pred_t = [idx for idx, x in enumerate(pred_score) if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)
    return person_boxes

def get_pose_estimation_prediction(pose_model, image, center, scale):
    """
    Perform pose estimation on the image given the center and scale.
    Returns the predicted keypoints.
    """
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.INPUT_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.INPUT_SIZE[1]), int(cfg.MODEL.INPUT_SIZE[0])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    model_input = transform(model_input).unsqueeze(0).to(CTX)
    pose_model.eval()
    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))
        return preds

def parse_args():
    parser = argparse.ArgumentParser(description='Real-Time Pose Estimation for Knee Rehabilitation')
    parser.add_argument('--cfg', type=str, default='../config/my_hrnet_config.yaml', help='Path to config file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for inference')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--write', action='store_true', help='Save output video/image')
    parser.add_argument('--showFps', action='store_true', help='Display FPS on output')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def main():
    # CUDNN related settings
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    # Path to your trained pose estimation model
    TRAINED_MODEL_PATH = '../models/final_state.pth'

    # Initialize object detection model (Faster R-CNN)
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    box_model.to(CTX)
    box_model.eval()

    # Initialize pose estimation model
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    # Load your trained model weights
    if os.path.exists(TRAINED_MODEL_PATH):
        print('=> loading trained model from {}'.format(TRAINED_MODEL_PATH))
        try:
            pose_model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=CTX), strict=False)
        except TypeError:
            # If weights_only is not supported, fallback
            print("weights_only=True not supported, loading with weights_only=False")
            pose_model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=CTX), strict=False)
    else:
        print('Error: Trained model file not found at {}'.format(TRAINED_MODEL_PATH))
        return

    # Move model to the appropriate device and set to eval mode
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Initialize SquatDetector
    squat_detector = SquatDetector()

    # Initialize Video Capture
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
        if not vidcap.isOpened():
            print("Error: Cannot open webcam.")
            return
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
        if not vidcap.isOpened():
            print(f"Error: Cannot open video file {args.video}.")
            return
    else:
        print('Please use --video or --webcam to define the input.')
        return

    # Initialize Video Writer if needed
    if args.write:
        save_path = 'output.avi' if args.webcam else 'output_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, fourcc, 24.0, (frame_width, frame_height))
        print(f"Saving output to {save_path}")

    # Initialize variables for FPS calculation
    prev_time = 0

    while True:
        ret, image_bgr = vidcap.read()
        if not ret:
            print('Cannot load the video frame.')
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        image = image_bgr[:, :, [2, 1, 0]]

        input_list = []
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().to(CTX)
        input_list.append(img_tensor)

        # Object detection to get person bounding boxes
        pred_boxes = get_person_detection_boxes(box_model, input_list, threshold=0.9)

        # Initialize feedback messages
        feedback_messages = []

        # Pose estimation and analysis
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.INPUT_SIZE[0], cfg.MODEL.INPUT_SIZE[1])
                pose_preds = get_pose_estimation_prediction(pose_model, image_bgr, center, scale)
                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        # Draw skeleton on the image
                        draw_pose(kpt, image_bgr)

                        # Calculate knee angles
                        left_knee_angle = calculate_angle(kpt[11], kpt[13], kpt[15])
                        right_knee_angle = calculate_angle(kpt[12], kpt[14], kpt[16])

                        # Update SquatDetector with current angles
                        feedback = squat_detector.update(left_knee_angle, right_knee_angle)
                        feedback_messages.extend(feedback)

                        # Display angles
                        cv2.putText(image_bgr, f"Left Knee: {int(left_knee_angle)}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(image_bgr, f"Right Knee: {int(right_knee_angle)}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        # Provide feedback based on angles
                        if left_knee_angle < 90:
                            cv2.putText(image_bgr, "Left Knee Too Bent", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        if right_knee_angle < 90:
                            cv2.putText(image_bgr, "Right Knee Too Bent", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display squat count and feedback
        cv2.putText(image_bgr, f"Squats: {squat_detector.count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for idx, msg in enumerate(feedback_messages):
            cv2.putText(image_bgr, msg, (10, 180 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display FPS
        if args.showFps:
            cv2.putText(image_bgr, f'FPS: {fps:.2f}', (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Write the frame to the output video if needed
        if args.write:
            out.write(image_bgr)

        # Show the image
        cv2.imshow('Knee Rehab Check', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    vidcap.release()
    if args.write:
        out.release()
        print('Video has been saved as {}'.format(save_path))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

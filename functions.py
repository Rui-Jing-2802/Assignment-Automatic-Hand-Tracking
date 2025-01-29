import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision    

# def filter_landmarks_by_depth(landmarks, z_values, max_depth=None):
#     """
#     filter landmarks by the distance to the samera (Z value）
#     - landmarks: List[(x, y)]，coordinate for the landmark
#     - z_values: List[float]，the z-value for each landmark
#     - max_depth: float，The threshold of the Z value
#     """
#     if max_depth is not None:
#         filtered_landmarks = [
#             (x, y) for (x, y), z in zip(landmarks, z_values) if z <= max_depth
#         ]
#     else:
#         filtered_landmarks = landmarks  
#     return filtered_landmarks


# def detect_hand_landmarks(image_path, detector):
#     """
#     
#     """
#     image = mp.Image.create_from_file(image_path)
#     detection_result = detector.detect(image)
#     if detection_result.hand_landmarks:
#         landmarks_with_depth = [
#             ([(landmark.x, landmark.y) for landmark in hand_landmarks],
#              [landmark.z for landmark in hand_landmarks])  # added z value
#             for hand_landmarks in detection_result.hand_landmarks
#         ]
#         return landmarks_with_depth  # return both landmarks and Z value
#     return None


def detect_hand_landmarks(image_path, detector):
    """
    Detects hand landmarks for all hands in an image.
    """
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    if detection_result.hand_landmarks:
        return [
            [(landmark.x, landmark.y) for landmark in hand_landmarks]
            for hand_landmarks in detection_result.hand_landmarks
        ]  
    return None

# def filter_key_landmarks(landmarks):
#     """
#     filter key landmarks
#     """
#     key_indices = [] 
#     return [landmarks[i] for i in key_indices if i < len(landmarks)]

def get_hand_box(landmarks, width, height):
    """
    Using landmarks to create bounding boxes
    """
    x_coords = [int(x * width) for x, y in landmarks]
    y_coords = [int(y * height) for x, y in landmarks]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]  # [x_min, y_min, x_max, y_max]


def generate_mask_with_sam2(frame_idx, frame_path, all_landmarks, predictor, inference_state):
    """
    Generates a combined segmentation mask for all hands using SAM2.
    """
    image = mp.Image.create_from_file(frame_path)
    height, width, _ = image.numpy_view().shape

    combined_mask = None
    for landmarks in all_landmarks:
        key_landmarks = landmarks
        points = np.array([[int(x * width), int(y * height)] for x, y in key_landmarks], dtype=np.float32)

        hand_box = get_hand_box(landmarks, width, height)

        # Using negative points to avoid mask on the arm
        negative_points = [
            (400, 400),  # right arm
            (900, 600),  # left arm
        ]

        # merge both positive and negative clicks
        all_points = np.vstack((points, negative_points))
        all_labels = np.array([1] * len(points) + [0] * len(negative_points), dtype=np.int32)


        _, _, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=all_points,
            labels=all_labels,
            box = hand_box
        )

        mask = (mask_logits > 0).cpu().numpy()
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.logical_or(combined_mask, mask)  # Combine masks

    return combined_mask


def get_mask_image(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype('uint8')

    if len(mask_image.shape) == 2:  # Single channel
        mask_image = cv2.merge([mask_image, mask_image, mask_image])  # Convert to 3 channels
    elif mask_image.shape[2] == 4:  # 4-channel (e.g., RGBA)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)

    return mask_image


def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Generate the final video out of the frames
    """
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.jpeg'))],
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )

    # Using the first frame as an example to get the frame's shape
    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frames:
        video_writer.write(cv2.imread(frame_path))
    video_writer.release()


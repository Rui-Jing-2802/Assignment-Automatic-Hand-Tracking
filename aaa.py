import cv2
import mediapipe as mp
import os
import numpy as np

import torch
from sam2.build_sam import build_sam2_video_predictor

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# Initialize MediaPipe HandLandmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Process frames and extract bounding boxes
def detect_hands_in_frame(frame_path):
    frame = cv2.imread(frame_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        bounding_boxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            h, w, _ = frame.shape
            bounding_box = [int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)]
            bounding_boxes.append(bounding_box)
        return bounding_boxes
    return []

# # Example usage
video_dir = "./data/test_JPEG"
frames = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')])

hand_bboxes = {}
for frame_path in frames:
    frame_name = os.path.basename(frame_path)
    hand_bboxes[frame_name] = detect_hands_in_frame(frame_path)




# Load SAM2 predictor
sam2_checkpoint = "./sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Initialize SAM2 inference state
inference_state = predictor.init_state(video_path=video_dir)

# Process frames with SAM2
for frame_idx, frame_path in enumerate(frames):
    frame_name = os.path.basename(frame_path)
    for bbox in hand_bboxes.get(frame_name, []):
        bbox_array = np.array(bbox, dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,  # Unique ID for hands
            box=bbox_array
        )

# Propagate masks across the video
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Create output video with masks
# output_video_path = "./output_video.mp4"
# frame_height, frame_width, _ = cv2.imread(frames[0]).shape
# out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# for frame_idx, frame_path in enumerate(frames):
#     frame = cv2.imread(frame_path)
#     if frame_idx in video_segments:
#         for mask in video_segments[frame_idx].values():
#             mask = cv2.resize(mask, (frame_width, frame_height)).astype(np.uint8) * 255
#             frame[mask > 0] = [0, 0, 255]  # Highlight hands in red
#     out_video.write(frame)

# out_video.release()


output_frames_dir = "./output_frames"
os.makedirs(output_frames_dir, exist_ok=True)

for frame_idx, frame_path in enumerate(frames):
    frame = cv2.imread(frame_path)
    if frame_idx in video_segments:
        for mask in video_segments[frame_idx].values():
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8) * 255
            # Apply the mask color (e.g., red for hands)
            frame[mask > 0] = [0, 0, 255]  # Red for hands
    # Save the frame with masks
    output_frame_path = os.path.join(output_frames_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(output_frame_path, frame)

print(f"Output frames saved in: {output_frames_dir}")
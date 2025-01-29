import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2

from functions import show_mask
from functions import show_points
from functions import show_box

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision    

device = "cuda"

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







sam2_checkpoint = "./sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Load JPEG frames obtained from test.mp4
video_dir = "./data/test_JPEG"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# stores all JPEG frames' pixels in the inference_state
inference_state = predictor.init_state(video_path=video_dir)




# Initialize hand_landmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def detect_hand_landmarks(image_path):
    """
    Detects hand landmarks for all hands in an image.
    """
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    if detection_result.hand_landmarks:
        return [
            [(landmark.x, landmark.y) for landmark in hand_landmarks]
            for hand_landmarks in detection_result.hand_landmarks
        ]  # Return landmarks for all hands
    return None



def generate_mask_with_sam2(frame_idx, frame_path, all_landmarks):
    """
    Generates a combined segmentation mask for all hands using SAM2.
    """
    image = mp.Image.create_from_file(frame_path)
    height, width, _ = image.numpy_view().shape

    combined_mask = None
    for landmarks in all_landmarks:
        points = np.array([[int(x * width), int(y * height)] for x, y in landmarks], dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)  # Positive clicks

        _, _, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
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

output_frames_dir = "./output_frames"
os.makedirs(output_frames_dir, exist_ok=True)

for idx, frame_name in enumerate(frame_names):
    frame_path = os.path.join(video_dir, frame_name)
    landmarks = detect_hand_landmarks(frame_path)
    mask = generate_mask_with_sam2(idx, frame_path, landmarks)
    if mask is not None:
        frame = cv2.imread(frame_path)
        mask_image = get_mask_image(mask, plt.gca())
        
        # print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
        # print(f"Mask image shape: {mask_image.shape if mask_image is not None else 'None'}")
        # print(f"Frame dtype: {frame.dtype}")
        # print(f"Mask image dtype: {mask_image.dtype}")
        mask_overlay = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
        cv2.imwrite(os.path.join(output_frames_dir, frame_name), mask_overlay)


# def create_video_from_frames(frames_dir, output_path, fps=30):
#     frames = sorted(
#         [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.jpeg'))],
#         key=lambda x: int(os.path.basename(x).split('.')[0])
#     )
#     frame = cv2.imread(frames[0])
#     height, width, _ = frame.shape

#     fourcc = cv2.cv.VideoWriter_fourcc(*'mp4v')  # Alternatively, try 'XVID'
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     for frame_path in frames:
#         video_writer.write(cv2.imread(frame_path))
#     video_writer.release()

# create_video_from_frames(output_frames_dir, "./output_video.mp4")

import os
import cv2
from PIL import Image
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

from functions import detect_hand_landmarks
from functions import generate_mask_with_sam2
from functions import get_mask_image
from functions import create_video_from_frames

def masking_hand(video_dir):
    """
    Takes directory for the video frames as input and generate video with mask on hand
    """
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


    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

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

    output_frames_dir = "./output_frames"
    os.makedirs(output_frames_dir, exist_ok=True)

    for idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(video_dir, frame_name)
        landmarks = detect_hand_landmarks(frame_path, detector)
        mask = generate_mask_with_sam2(idx, frame_path, landmarks, predictor, inference_state)
        if mask is not None:
            frame = cv2.imread(frame_path)
            mask_image = get_mask_image(mask, plt.gca())
            
            mask_overlay = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            cv2.imwrite(os.path.join(output_frames_dir, frame_name), mask_overlay)

    create_video_from_frames("./output_frames", "./output_video.mp4")


def main():
    video_dir = "./data/test_JPEG"
    masking_hand(video_dir)

if __name__=='__main__':
    main()
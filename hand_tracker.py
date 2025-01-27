import os
import cv2

from hand_track import draw_landmarks_on_image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
        
# Load JPEG frames obtained from test.mp4
video_dir = "./data/test_JPEG"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# STEP 1: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


# # Loop through all the frames in the directory
# for frame_name in frame_names:
#     # Construct the full path to the frame
#     frame_path = os.path.join(video_dir, frame_name)
    
#     # STEP 3: Load the frame as an image for MediaPipe
#     image = mp.Image.create_from_file(frame_path)
    
#     # STEP 4: Detect hand landmarks from the frame
#     detection_result = detector.detect(image)
    
#     # STEP 5: Visualize the detection result
#     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
#     # Show the annotated image
#     cv2.imshow('Annotated Frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
#     # Wait for a key press to proceed to the next frame (press any key to continue or 'q' to quit)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# STEP 2: Load the input image.
first_frame_path = os.path.join(video_dir, frame_names[0])

image = mp.Image.create_from_file(first_frame_path)

# STEP 3: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 4: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("hello", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
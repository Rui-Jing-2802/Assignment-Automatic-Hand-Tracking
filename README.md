
# Automatic-Hand-Tracking

## Goal
The goal for this project is to build an automatic pipeline that tracks hand movements in a video using SAM2 and Mediapipe's Hand Landmarker. This project focuses on extracting precise masks of hands from video frames to enable downstream tasks such as gesture recognition, interaction modeling, or hand pose estimation.


## Set Up

1. Download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

2. To ensure reproducibility, you can configure the Python environment using the provided `requirements.txt` file.
 

## Work Pipeline

1.  **Data-preprocessing**:
	- The individual frames were extracted from the video and stored in the path ./data/test_JPEG/ using the `ffmpeg` command-line tool:
```bash
ffmpeg -i  input_video.mp4  -q:v  2  -start_number  0  output_frames/%05d.jpg
```
2.  **Hand Detection and Prompt Generation**:
    References: [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), [Google Colab Sample](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
    -   Mediapipe Hand Landmarker detects 21 hand landmarks per hand.
    -   Key landmarks (e.g., wrist, fingertips) are filtered and used as positive click prompts.
    -   Negative click prompts are placed in non-hand regions (e.g., background) to prevent mask spillovers.

3.  **Mask Creation with SAM2**:
   
    -   Bounding boxes around hand landmarks are generated to constrain segmentation regions.
    -   Click prompts and bounding boxes are fed into SAM2 to create masks for the hands in each frame.
    
4.  **Post-Processing**:
    
    -   Stored individual frames in ./frame_outputs/ and merge the frames into the final video:

https://github.com/user-attachments/assets/800245ef-47fc-404f-8646-78aca5f6b057




## Future Work
### Problems unsolved
The generated video is not yet perfect with the following problems unsolved:
1. Frame 27 in the output have mask on the face and arm
2. Frame 176 to 187 have the right hand not fully masked, most likely caused because two hands folded
3. The masks in the generated video are rough and imprecise, with the problems of rough edges (the edges of the masks are jagged and do not accurately align with the hand contours) and incomplete coverage (certain parts of the hands, especially around the fingers and wrists, are not fully covered by the masks)

### Potential solutions
1. Filter key landmarks to reduce the excessive clicks
2. Using z coordinate of landmarks, which represents the landmark depth, with the depth at the wrist being the origin, to better handle hand overlaps.
3. Creating multiple sub-masks to divide hand and arm

## Citation

If you find **SAM2Point** useful for your research or applications, please kindly cite using this BibTeX:

```latex
@article{guo2024sam2point,
  title={SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners},
  author={Guo, Ziyu and Zhang, Renrui and Zhu, Xiangyang and Tong, Chengzhuo and Gao, Peng and Li, Chunyuan and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2408.16768},
  year={2024}
}
```


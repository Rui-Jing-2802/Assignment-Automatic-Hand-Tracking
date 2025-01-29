
# Assignment-Automatic-Hand-Tracking

  

## Data Pre-processing

  

The individual frames were extracted from the video and stored in the path ./data/test_JPEG/ using the `ffmpeg` command-line tool:
```bash
ffmpeg -i  input_video.mp4  -q:v  2  -start_number  0  output_frames/%05d.jpg
```

  

## Set Up

  

1. download a model checkpoint. All the model checkpoints can be downloaded by running:

  

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
 
## Future Work
### Problems unsolved
1. Frame 27 in the output have mask on the face and arm
2. Frame 176 to 187 have the right hand not fully masked, most likely caused because two hands folded
3. The masks in the generated video are rough and imprecise, with the problems of rough edges (the edges of the masks are jagged and do not accurately align with the hand contours) and incomplete coverage (certain parts of the hands, especially around the fingers and wrists, are not fully covered by the masks)

### Potential solutions
1. Filter key landmarks to reduce the amount of clicks
2. 

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


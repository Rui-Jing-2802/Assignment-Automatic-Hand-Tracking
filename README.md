# Assignment-Automatic-Hand-Tracking

# Converting Video to Frames

To handle the issue of extracting individual frames from the video, I used the `ffmpeg` command-line tool:

```bash
ffmpeg -i input_video.mp4 -q:v 2 -start_number 0 output_frames/%05d.jpg
```
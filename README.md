# Object-Detection-in-Video-using-PyTorch

This project implements a pipeline for detecting vehicles and other objects in video frames using PyTorch and OpenCV. The notebook processes a video, extracts individual frames, applies an object detection model to identify vehicles, and stitches the processed frames back into a video with bounding boxes drawn around detected objects.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Key Components](#key-components)
5. [Model Inference](#model-inference)
6. [Output](#output)
7. [Sample Videos](#sample-videos)

## Overview

The notebook extracts frames from a given video file and performs vehicle and object detection on the frames. Once the objects are detected and bounding boxes are added, the processed frames are recompiled into a video file.

Key steps in the notebook:
- Extract frames from the video.
- Use a pre-trained deep learning model for object detection.
- Draw bounding boxes around detected vehicles and objects.
- Stitch the processed frames back into a video.

## Installation

To run the code, you will need the following dependencies:

```bash
pip install torch torchvision opencv-python
```

### Clone this repository

```bash
git clone https://github.com/yourusername/Vehicle-and-Object-Detection-in-Videos.git
cd Vehicle-and-Object-Detection-in-Videos
```

## Usage

1. Place the video you want to process in the project directory.
2. Modify the `video_path` variable in the notebook to point to the video file.
3. Run the notebook to extract frames, perform object detection, and stitch the processed frames back into a video.

Example code snippet:

```python
import cv2
import os

video_path = 'your_video.mp4'
vidcap = cv2.VideoCapture(video_path)

frames_dir = './frames/'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

success, image = vidcap.read()
count = 0
while success:
    frame_filename = os.path.join(frames_dir, f"frame{count}.jpg")
    cv2.imwrite(frame_filename, image)
    success, image = vidcap.read()
    count += 1

vidcap.release()
```

### Stitching the Frames into a Video

After processing the frames, the notebook also stitches them back into a video file. The output video will have bounding boxes drawn around the detected objects.

Example stitching code:

```python
frame_array = []
for count in range(len(frames)):
    frame = cv2.imread(f'./frames/frame{count}.jpg')
    height, width, layers = frame.shape
    size = (width, height)
    frame_array.append(frame)

output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for frame in frame_array:
    output_video.write(frame)

output_video.release()
```

## Key Components

1. **Frame Extraction**: The notebook uses OpenCV to read and extract frames from a video file.
2. **Object Detection Model**: A pre-trained object detection model is applied to each frame to detect vehicles and other objects.
3. **Bounding Boxes**: For each detected object, a bounding box is drawn around it, and the resulting frames are saved.
4. **Video Stitching**: After object detection, the processed frames are compiled back into a video using OpenCV's `VideoWriter`.

## Model Inference

The object detection model is applied to each extracted frame. You can modify the model section of the code to use any other object detection architecture or pre-trained weights from PyTorch's `torchvision` models.

Example model usage:

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Apply the model to each frame for inference
```

## Output

The processed frames with bounding boxes are saved in a specified directory. Additionally, these frames are stitched back into a video file, and the final video output with detected objects and bounding boxes is saved as `output_video.avi`.

## Sample Videos

You can find the input and output videos at the following links:

- [Download input video](https://drive.google.com/file/d/1ktu9_TXndgGVFw9jhG9HHIwuBlqIjZXy/view?usp=drive_link)
- [Download output video](https://drive.google.com/file/d/11xtYbSM2tJsa1hLfUDVBjJJKzWRh6pKC/view?usp=drive_link)

# Real-Time Hand and Face Tracking with MediaPipe and OpenCV

This repository contains two Python scripts demonstrating real-time computer vision applications using [MediaPipe](https://google.github.io/mediapipe/) and [OpenCV](https://opencv.org/):

- **Hand Gesture Recognition:** Detects and counts fingers on one or multiple hands using MediaPipe Hands.
- **Face Mesh Detection:** Tracks detailed 3D facial landmarks using MediaPipe Face Mesh.

---

## Features

- **Accurate hand landmark detection** with finger counting logic.
- **Detailed face mesh visualization** with over 400 facial landmarks.
- Real-time processing using webcam video feed.
- Easy to run scripts with minimal dependencies.

---

## Installation

Ensure you have Python 3.7+ installed. Then install the required packages via pip:

```bash
pip install opencv-python mediapipe numpy

Usage
Run Hand Gesture Recognition
python face_mesh.py

Run Face Mesh Detection
python face_mesh.py

Press q in the video window to quit


How It Works
The scripts capture video from your webcam.

MediaPipe processes each frame for hand or face landmarks.

Landmarks are drawn directly on the frames and displayed in a window.

For hand gestures, finger counting is performed based on landmark positions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Thanks to the MediaPipe team for providing amazing real-time ML solutions.

Built with OpenCV for computer vision processing.
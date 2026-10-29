# Simple Face Detector

Real-time face detection from your webcam in under 70 lines of Python and OpenCV. Each detected face gets a green box, and a live counter shows how many faces are in frame.

## How it works

It uses a **Haar cascade**, the classic Viola–Jones detector that ships with OpenCV. Instead of a neural network, it slides a window over the image and checks simple light/dark rectangle patterns (like "the eye region is darker than the cheeks") in a fast chain of stages. Most windows get rejected in the first few stages, which is why it runs in real time on a CPU.

For each webcam frame, the script:

1. Converts the frame to grayscale, since the cascade only looks at brightness.
2. Runs `detectMultiScale`, which searches at several sizes so near and far faces are both found.
3. Draws a box around each hit and overlays the face count.

Two parameters control the trade-off between catching faces and avoiding false alarms:

| Parameter | Value | Effect |
| --- | --- | --- |
| `scaleFactor` | 1.1 | Image shrinks 10% per search pass. Lower finds more sizes but runs slower. |
| `minNeighbors` | 5 | Overlapping hits needed to count as a face. Higher means fewer false positives. |

## Run it

```bash
git clone https://github.com/mercyy1/Simple-Face-Detector.git
cd Simple-Face-Detector
pip install -r requirements.txt
python face_detector.py            # default webcam
python face_detector.py --camera 1 # a different camera
```

Press **q** or **Esc** to quit.

## Limitations

Haar cascades are fast but dated. They struggle with faces turned sideways, poor lighting, and partial occlusion. For better accuracy, swap in a deep-learning detector such as OpenCV's DNN face detector or MediaPipe Face Detection.

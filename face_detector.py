"""Real-time face detection from a webcam using OpenCV's Haar cascade.

Usage:
    python face_detector.py            # default webcam
    python face_detector.py --camera 1 # pick another camera

Press q (or Esc) in the video window to quit.
"""
import argparse
import sys

import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam face detection")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    args = parser.parse_args()

    # Load the pre-trained frontal-face Haar cascade that ships with OpenCV.
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Could not load Haar cascade from {CASCADE_PATH}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Is it connected and not in use by another app?",
              file=sys.stderr)
        return 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame", file=sys.stderr)
                break

            # Haar cascades work on intensity, so convert to grayscale first.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # scaleFactor: how much the image shrinks per search pass (smaller = slower, finds more sizes)
            # minNeighbors: how many overlapping hits a face needs (higher = fewer false positives)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Face Detector - press q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
    finally:
        # Always release the camera, even if something above throws.
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

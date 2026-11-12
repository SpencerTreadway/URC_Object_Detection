#!/usr/bin/env python3
"""Run a trained YOLO detector continuously on a camera stream."""

import argparse

import cv2
from ultralytics import YOLO

from camera_init import open_camera

def parse_source(value: str) -> int | str:
    """Treat numeric sources as camera indices and other values as URLs."""

    try:
        return int(value)
    except ValueError:
        return value

def run(
    model_path: str,
    source: int | str,
    confidence: float,
    image_size: int,
    device: str | None,
) -> None:
    """Capture frames and draw detections until q or Escape is pressed."""

    # load the YOLO model and open the camera stream
    model = YOLO(model_path)
    camera = open_camera(source)
    try:
        while True:
            # read a frame from the camera and raise an error if it fails
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            # run the YOLO model on the frame and draw the detections
            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )
            # draw the detections on the frame and display it in a window
            annotated_frame = results[0].plot()
            cv2.imshow("Object detection", annotated_frame)

            # wait for a key press and exit if q or Escape is pressed
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        # release the camera and close the window
        camera.release()
        cv2.destroyAllWindows()

# main function to parse command-line arguments and run the object detector
def main() -> None:
    # parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained weights, for example runs/detect/orange_hammer/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or stream URL (default: 0)",
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as cpu, mps, or 0 (default: Ultralytics auto-select)",
    )
    args = parser.parse_args()
    # run the object detector on the camera stream
    run(
        args.model,
        parse_source(args.source),
        args.conf,
        args.imgsz,
        args.device,
    )

# run the main function if this script is executed directly
if __name__ == "__main__":
    main()

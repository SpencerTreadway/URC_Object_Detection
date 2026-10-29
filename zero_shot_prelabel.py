#!/usr/bin/env python3
"""Zero-shot pre-labeling of a folder of images with OWLv2 (text-prompted object detection).

For every image it proposes bounding boxes for the object you describe in text, and writes:

    OUT_DIR/yolo/<image>.txt                  YOLO-format labels (class 0). Empty file = no detection.
    OUT_DIR/labelstudio_predictions.json      Label Studio tasks with pre-annotations (import this).
    OUT_DIR/preview/<image>.jpg               image with the proposed boxes drawn, for quick review
    OUT_DIR/summary.csv                       per-image detection count, top score, and a review flag

Example:
    python prelabel_owlv2.py ~/dataset/test_run/rgb ~/dataset/test_run/prelabels \
        --ls-prefix test_run/rgb --label-name mallet

How the false-positive suppression works: besides the object you want ("positive" prompts), you
give "distractor" prompts for things that look similar (an orange screwdriver, a metal ruler...).
Every candidate box is scored against ALL prompts, and it is kept only if a positive prompt is its
best match. So an orange screwdriver handle that matches "orange screwdriver" better than
"orange rubber mallet" is dropped.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

# Lets PyTorch fall back to CPU for any op the Apple GPU (MPS) doesn't support.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

# constants and default prompts for zero-shot pre-labeling with OWLv2.
IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_POSITIVE = ["a photo of an orange rubber mallet"]
DEFAULT_DISTRACTORS = [
    "a photo of an orange screwdriver",
    "a photo of a metal ruler",
    "a photo of a laptop",
    "a photo of a wooden table",
]


# ----------------------------------------------------------------------------------------------
# Pure-numpy post-processing (no model needed, so it can be unit-tested)
# ----------------------------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# Non-maximum suppression (NMS) for filtering overlapping boxes.
def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy non-maximum suppression. Returns indices to keep, best score first."""
    # Sort boxes by score in descending order and apply greedy NMS.
    order = np.argsort(-scores)
    keep = []
    # Keep the box with the highest score and remove all overlapping boxes above the IoU threshold.
    while order.size:
        i = order[0]
        # Append the index of the current highest-scoring box to the keep list.
        keep.append(int(i))
        if order.size == 1:
            break
        # Compute the IoU of the current highest-scoring box with the rest.
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0]) # x-coordinate of the top-left corner of the intersection
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1]) # y-coordinate of the top-left corner of the intersection
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2]) # x-coordinate of the bottom-right corner of the intersection
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3]) # y-coordinate of the bottom-right corner of the intersection
        # Compute the area of the intersection rectangle.
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        # Compute the area of the current highest-scoring box and the rest.
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        # Compute the Intersection over Union (IoU) and filter out boxes with IoU above the threshold.
        iou = inter / (area_i + area_r - inter + 1e-9)
        # Keep only the boxes with IoU less than or equal to the threshold.
        order = rest[iou <= iou_thr]
    return keep

# Select detections from OWLv2 outputs based on score threshold, NMS, and area constraints.
def select_detections(
    logits: np.ndarray,
    boxes_cxcywh: np.ndarray,
    n_pos: int,
    size: tuple[int, int],
    thr: float,
    nms_iou: float = 0.5,
    max_det: int = 3,
    max_area_frac: float = 0.6,
) -> list[tuple[float, float, float, float, float]]:
    """Turn raw OWLv2 outputs into a short list of (x1, y1, x2, y2, score) in image pixels.

    logits:        (num_patches, num_queries) raw scores; the first n_pos queries are positives.
    boxes_cxcywh:  (num_patches, 4) boxes normalized to the PADDED SQUARE image (OWLv2 pads the
                   image to a square at the bottom/right), so pixel = normalized * max(W, H).
    """
    # Extract image width, height, and the side length of the padded square image.
    width, height = size
    side = float(max(width, height))

    # Convert logits to probabilities using the sigmoid function.
    scores = sigmoid(np.asarray(logits, dtype=np.float64))
    # Determine the best query and its corresponding score for each patch.
    best_q = scores.argmax(axis=1)
    # Get the score of the best query for each patch.
    best_s = scores[np.arange(scores.shape[0]), best_q]

    # Create a mask to keep only the positive prompts with scores above the threshold.
    keep_mask = (best_q < n_pos) & (best_s >= thr)  # a positive prompt must be the best match
    if not keep_mask.any():
        return []

    # Keep only the boxes corresponding to the positive prompts above the threshold and convert them to absolute coordinates.
    b = np.asarray(boxes_cxcywh, dtype=np.float64)[keep_mask] * side
    # Keep only the scores corresponding to the kept boxes.
    s = best_s[keep_mask]
    # Convert the boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    # Clip the coordinates to be within the image boundaries.
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)

    # Compute the width and height of the boxes and filter out boxes that are too small or too large.
    bw = xyxy[:, 2] - xyxy[:, 0]
    bh = xyxy[:, 3] - xyxy[:, 1]
    ok = (bw >= 2) & (bh >= 2) & ((bw * bh) <= max_area_frac * width * height)
    xyxy, s = xyxy[ok], s[ok]
    if len(s) == 0:
        return []

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes based on the IoU threshold.
    kept = nms(xyxy, s, nms_iou)[:max_det]
    return [(*map(float, xyxy[i]), float(s[i])) for i in kept]

# Convert a detection to a YOLO-formatted line.
def to_yolo_line(
    det: tuple[float, float, float, float, float],
    size: tuple[int, int],
    cls: int = 0,
) -> str:
    # Convert the detection from (x1, y1, x2, y2) format to YOLO format (x_center, y_center, width, height) normalized by the image size.
    x1, y1, x2, y2, _ = det # Unpack the detection tuple, ignoring the score for YOLO conversion.
    w, h = size
    # Compute the normalized center coordinates, width, and height for the YOLO format.
    return (f"{cls} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

# Convert a detection to a Label Studio task.
def to_labelstudio_task(
    image_url: str,
    size: tuple[int, int],
    dets: list[tuple[float, float, float, float, float]],
    label_name: str,
    from_name: str,
    to_name: str,
    model_version: str,
) -> dict[str, Any]:
    w, h = size
    results = []
    for (x1, y1, x2, y2, score) in dets:
        # Convert the detection coordinates to percentages relative to the image size for Label Studio.
        results.append({
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels",
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "score": round(score, 4),
            "value": {
                "x": 100.0 * x1 / w,
                "y": 100.0 * y1 / h,
                "width": 100.0 * (x2 - x1) / w,
                "height": 100.0 * (y2 - y1) / h,
                "rotation": 0,
                "rectanglelabels": [label_name],
            },
        })
    # Return the task dictionary containing the image URL and predictions (if any).
    task = {"data": {"image": image_url}}
    if results:
        task["predictions"] = [{
            "model_version": model_version,
            "score": round(max(d[4] for d in dets), 4),
            "result": results,
        }]
    return task

# Draw a preview of the detections on the image and save it to the specified path.
def draw_preview(image, dets, path: Path) -> None:
    # Draw rectangles and scores for each detection on a copy of the image and save it.
    from PIL import ImageDraw
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, score) in dets:
        # Draw the rectangle and the score for the current detection.
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        # Offset the text slightly from the top-left corner of the rectangle.
        draw.text((x1 + 4, max(0, y1 - 14)), f"{score:.2f}", fill=(0, 255, 0))
    img.save(path, quality=90)

# Load the manifest file containing extractor provenance keyed by the RGB path.
def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load extractor provenance keyed by the RGB path."""
    if not path.is_file():
        return {}
    with path.open(newline="") as file:
        return {row["rgb_file"]: row for row in csv.DictReader(file)}

# ----------------------------------------------------------------------------------------------
# Model + main
# ----------------------------------------------------------------------------------------------
# Pick the appropriate device (CPU, CUDA, or MPS) for model inference.
def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Main entry point for the script.
def main() -> int:
    # Parse command-line arguments and validate them.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, help="RGB image folder, e.g. frames/s00/rgb")
    ap.add_argument("out_dir", type=Path, help="pre-label output folder")
    ap.add_argument("--prompt", action="append",
                    help="text describing the target object; repeat for synonyms "
                         f"(default: {DEFAULT_POSITIVE[0]!r})")
    ap.add_argument("--distractor", action="append",
                    help="text for look-alike things to suppress; repeat for several "
                         "(default: screwdriver, ruler, laptop, table)")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    ap.add_argument("--threshold", type=float, default=0.2, help="minimum score to propose a box")
    ap.add_argument("--review-below", type=float, default=0.4,
                    help="flag images whose best score is below this (or that have no box) for review")
    ap.add_argument("--max-det", type=int, default=3, help="max boxes kept per image")
    ap.add_argument("--label-name", default="mallet",
                    help="class name; must match your Label Studio label EXACTLY")
    ap.add_argument("--from-name", default="label", help="Label Studio labeling-config from_name")
    ap.add_argument("--to-name", default="image", help="Label Studio labeling-config to_name")
    ap.add_argument("--ls-prefix", default="",
                    help="image folder path RELATIVE to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "
                         "(e.g. test_run/rgb)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (for a test run)")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="extractor manifest.csv (default: parent folder's manifest.csv when present)",
    )
    args = ap.parse_args()

    if not 0 <= args.threshold <= 1:
        ap.error("--threshold must be between 0 and 1")
    if not 0 <= args.review_below <= 1:
        ap.error("--review-below must be between 0 and 1")
    if args.max_det < 1:
        ap.error("--max-det must be at least 1")
    if args.limit < 0:
        ap.error("--limit cannot be negative")

    positives = args.prompt or DEFAULT_POSITIVE
    distractors = args.distractor if args.distractor is not None else DEFAULT_DISTRACTORS
    queries = positives + distractors

    img_dir = args.images.expanduser()
    if not img_dir.is_dir():
        ap.error(f"not a folder: {img_dir}")
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        ap.error(f"no images found in {img_dir}")

    out = args.out_dir.expanduser()
    for sub in ("yolo", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.expanduser()
        if args.manifest is not None
        else img_dir.parent / "manifest.csv"
    )
    manifest = load_manifest(manifest_path)

    # Import necessary libraries for model inference and image processing.
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    # Pick the device for model inference (CPU, CUDA, or MPS).
    device = pick_device(args.device)
    print(f"Loading {args.model} on {device} (first run downloads the weights)...")
    # Load the processor and model for object detection.
    processor = Owlv2Processor.from_pretrained(args.model)
    # Move the model to the selected device and set it to evaluation mode.
    model = Owlv2ForObjectDetection.from_pretrained(args.model).to(device).eval()
    # Print the queries for reference.
    print(f"Queries: positives={positives}  distractors={distractors}")

    # Initialize lists to store Label Studio tasks and summary rows.
    tasks, rows = [], []
    # Prepare the prefix for local file URLs in Label Studio tasks.
    prefix = args.ls_prefix.strip("/")
    # Iterate over each image path and perform object detection.
    for n, path in enumerate(paths, 1):
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
        # Get the size of the image (width, height).
        size = image.size  # (W, H)

        # Prepare the inputs for the model and perform inference.
        inputs = processor(text=[queries], images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        # Extract logits and predicted bounding boxes from the model outputs.
        logits = outputs.logits[0].float().cpu().numpy()          # (patches, queries)
        boxes = outputs.pred_boxes[0].float().cpu().numpy()       # (patches, 4) cx,cy,w,h

        # Select the final detections based on the logits, bounding boxes, and threshold.
        dets = select_detections(logits, boxes, len(positives), size, args.threshold,
                                 max_det=args.max_det)

        # Save the detections in YOLO format and draw a preview image.
        (out / "yolo" / f"{path.stem}.txt").write_text(
            "\n".join(to_yolo_line(d, size) for d in dets) + ("\n" if dets else ""))
        # Draw a preview image with the detected bounding boxes.
        draw_preview(image, dets, out / "preview" / f"{path.stem}.jpg")

        # Prepare the URL for the image and create a Label Studio task.
        url = f"/data/local-files/?d={prefix + '/' if prefix else ''}{path.name}"
        tasks.append(
            to_labelstudio_task(
                url,
                size,
                dets,
                args.label_name,
                args.from_name,
                args.to_name,
                args.model,
            )
        )

        # Determine if the image needs review based on detection scores and count.
        top = max((d[4] for d in dets), default=0.0)
        review = (not dets) or top < args.review_below or len(dets) > 1
        provenance = manifest.get(f"rgb/{path.name}", {})
        rows.append(
            [
                path.name,
                provenance.get("color_timestamp_ns", ""),
                provenance.get("depth_timestamp_ns", ""),
                provenance.get("timestamp_delta_ms", ""),
                len(dets),
                f"{top:.3f}",
                "REVIEW" if review else "ok",
            ]
        )
        print(f"[{n}/{len(paths)}] {path.name}: {len(dets)} box(es), top score {top:.2f}")

    # Save the Label Studio tasks and summary CSV, and print a final summary.
    with (out / "labelstudio_predictions.json").open("w") as f:
        json.dump(tasks, f, indent=2)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "color_timestamp_ns",
                "depth_timestamp_ns",
                "timestamp_delta_ms",
                "n_boxes",
                "top_score",
                "flag",
            ]
        )
        w.writerows(rows)

    # Count the number of images flagged for review and print the final summary.
    n_review = sum(r[6] == "REVIEW" for r in rows)
    print(f"\nDone. {len(rows)} images, {n_review} flagged for closer review. Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
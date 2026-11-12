#!/usr/bin/env python3
"""Prepare a YOLO dataset and train an object detector.

The source dataset is expected to contain ``images/``, ``labels/``, and
``classes.txt``. Labels use the standard YOLO format:
``class_id x_center y_center width height``.

Example:
    python object_detection_train.py \
        --dataset datasets/orange_hammer_dataset_01 \
        --epochs 50
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# constant file extensions for images that we will accept
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 
def read_classes(path: Path) -> list[str]:
    """Read one class name per line, ignoring blank lines."""

    # read the file, strip whitespace, and ignore blank lines
    classes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No class names found in {path}")
    return classes


def image_label_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Return image/label pairs and fail if any label is missing."""

    # find all image and label files
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # find corresponding label files and ensure they exist
    pairs = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        pairs.append((image_path, label_path))
    return pairs


def link_or_copy(source: Path, destination: Path) -> None:
    """Create a hard link where possible, falling back to a normal copy."""

    # ensure the destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # create a hard link to the source file
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

# prepare the dataset for training.
def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    validation_fraction: float,
    seed: int,
    rebuild: bool,
) -> Path:
    """Create the train/val directory layout expected by Ultralytics."""

    # validate the validation fraction
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    # check if the output directory already exists and handle accordingly
    if output_dir.exists() and not rebuild:
        # check if data.yaml exists in the output directory
        yaml_path = output_dir / "data.yaml"
        if yaml_path.is_file():
            return yaml_path
        raise FileExistsError(
            f"{output_dir} exists without data.yaml; use --rebuild to recreate it"
        )
    # create the output directory structure
    if rebuild and output_dir.exists():
        shutil.rmtree(output_dir)

    # read the class names from classes.txt
    classes = read_classes(source_dir / "classes.txt")
    # get the image/label pairs from the source directory
    pairs = image_label_pairs(source_dir)

    # get randomized training/validation seed
    random.Random(seed).shuffle(pairs)
    # determine the number of validation samples and split the pairs accordingly
    validation_count = max(1, round(len(pairs) * validation_fraction))
    # split the pairs into validation and training sets
    validation_pairs = pairs[:validation_count]
    training_pairs = pairs[validation_count:]

    # create the output directory structure for images and labels
    for split, split_pairs in (("train", training_pairs), ("val", validation_pairs)):
        for image_path, label_path in split_pairs:
            link_or_copy(image_path, output_dir / split / "images" / image_path.name)
            link_or_copy(label_path, output_dir / split / "labels" / label_path.name)

    # create the data.yaml file
    yaml_path = output_dir / "data.yaml"
    # write the data.yaml file with the dataset path, training and validation image directories, and class names
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(output_dir.resolve()))}",
                "train: train/images",
                "val: val/images",
                f"names: {json.dumps(classes)}",
                "",
            ]
        )
    )
    return yaml_path

# parse command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/orange_hammer_dataset_01"),
        help="Source dataset directory (default: datasets/orange_hammer_dataset_01)",
    )
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=None,
        help="Prepared dataset directory (default: <dataset>/yolo_dataset)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO model")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="orange_hammer")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate the prepared dataset split instead of reusing it",
    )
    return parser.parse_args()

# main function to prepare the dataset and train the YOLO model
def main() -> None:
    # parse command-line arguments
    args = parse_args()
    # prepare the dataset and get the path to the data.yaml file
    prepared_dir = args.prepared_dataset or args.dataset / "yolo_dataset"
    # prepare the dataset and get the path to the data.yaml file
    data_yaml = prepare_dataset(
        args.dataset,
        prepared_dir,
        args.validation_fraction,
        args.seed,
        args.rebuild,
    )
    # train the YOLO model using the prepared dataset
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )

# run the main function if this script is executed directly
if __name__ == "__main__":
    main()

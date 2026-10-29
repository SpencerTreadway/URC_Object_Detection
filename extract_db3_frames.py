"""Extract synchronized color and depth images from a ROS 2 RealSense bag.

The input is the SQLite ``.db3`` file produced by the ROS 2 recorder used for
the rover recordings.

Usage:
    python extract_db3_frames.py RECORDING OUT_DIR [--every 10]

The output contains:
    OUT_DIR/rgb/000000.png       Color image
    OUT_DIR/depth/000000.png     16-bit depth image in sensor units
    OUT_DIR/intrinsics.json      Camera intrinsics and depth metadata
    OUT_DIR/manifest.csv         Timestamp and source-frame mapping
"""

import argparse
import bisect
import csv
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data/compressed"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data/compressedDepth"
COLOR_INFO_TOPIC = "/device_0/sensor_1/Color_0/image/camera_info"
DEPTH_INFO_TOPIC = "/device_0/sensor_0/Depth_0/image/camera_info"
DEPTH_INFO_TOPIC_FALLBACK = "/device_0/sensor_0/Depth_0/camera_info"
DEPTH_UNITS_TOPIC = "/device_0/sensor_0/option/Depth_Units/value"


class CdrReader:
    """Read the little-endian ROS 2 CDR fields used by these messages."""

    def __init__(self, data: bytes):
        if data[:2] != b"\x00\x01":
            raise ValueError("unsupported ROS 2 CDR encoding")
        self.data = data
        self.offset = 4

    def align(self, boundary: int) -> None:
        self.offset = (self.offset + boundary - 1) & ~(boundary - 1)

    def read(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated ROS 2 CDR message")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def scalar(self, fmt: str) -> int | float:
        size = struct.calcsize(fmt)
        self.align(size)
        return struct.unpack("<" + fmt, self.read(size))[0]

    def string(self) -> str:
        length = int(self.scalar("I"))
        raw = self.read(length)
        if not raw or raw[-1] != 0:
            raise ValueError("invalid ROS 2 CDR string")
        return raw[:-1].decode("utf-8")

    def sequence(self, item_size: int) -> bytes:
        length = int(self.scalar("I"))
        self.align(item_size)
        return self.read(length * item_size)

    def sequence_values(self, fmt: str) -> list[int | float]:
        item_size = struct.calcsize(fmt)
        length = int(self.scalar("I"))
        return list(struct.unpack("<" + fmt * length, self.read(item_size * length)))


@dataclass(frozen=True)
class BagMessage:
    message_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class ImageMessage:
    message_id: int
    timestamp_ns: int
    image: np.ndarray


def parse_header(reader: CdrReader) -> int:
    seconds = int(reader.scalar("i"))
    nanoseconds = int(reader.scalar("I"))
    reader.string()
    return seconds * 1_000_000_000 + nanoseconds


def parse_compressed_image(data: bytes) -> tuple[int, str, bytes]:
    reader = CdrReader(data)
    timestamp_ns = parse_header(reader)
    format_name = reader.string()
    image_data = reader.sequence(1)
    return timestamp_ns, format_name, image_data


def decode_image(data: bytes, topic: str) -> np.ndarray:
    """Decode a ROS compressed image, including compressedDepth PNG payloads."""
    png_start = data.find(b"\x89PNG\r\n\x1a\n")
    if png_start < 0:
        raise ValueError(f"{topic} does not contain a PNG payload")

    image = cv2.imdecode(
        np.frombuffer(data[png_start:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None:
        raise ValueError(f"OpenCV could not decode an image from {topic}")
    return image


def read_topic_ids(database: sqlite3.Connection) -> dict[str, int]:
    rows = database.execute("SELECT id, name FROM topics").fetchall()
    topics = {name: topic_id for topic_id, name in rows}
    required = (COLOR_TOPIC, DEPTH_TOPIC)
    missing = [topic for topic in required if topic not in topics]
    if missing:
        raise RuntimeError("recording is missing required topic(s): " + ", ".join(missing))
    return topics


def read_messages(
    database: sqlite3.Connection,
    topic_id: int,
) -> list[BagMessage]:
    return [
        BagMessage(message_id, timestamp_ns)
        for message_id, timestamp_ns in database.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        )
    ]


def read_image(
    database: sqlite3.Connection,
    message: BagMessage,
    topic_id: int,
    topic_name: str,
) -> ImageMessage:
    row = database.execute(
        "SELECT data FROM messages WHERE id = ? AND topic_id = ?",
        (message.message_id, topic_id),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"message {message.message_id} disappeared from the bag")
    timestamp_ns, _format_name, compressed_data = parse_compressed_image(row[0])
    image = decode_image(compressed_data, topic_name)
    return ImageMessage(message.message_id, timestamp_ns, image)


def nearest_message(
    messages: list[BagMessage],
    timestamps: list[int],
    timestamp_ns: int,
) -> BagMessage:
    position = bisect.bisect_left(timestamps, timestamp_ns)
    candidates = []
    if position < len(messages):
        candidates.append(messages[position])
    if position:
        candidates.append(messages[position - 1])
    return min(candidates, key=lambda message: abs(message.timestamp_ns - timestamp_ns))


def parse_camera_info(data: bytes) -> dict[str, object]:
    reader = CdrReader(data)
    timestamp_ns = parse_header(reader)
    height = int(reader.scalar("I"))
    width = int(reader.scalar("I"))
    distortion_model = reader.string()
    distortion = reader.sequence_values("d")
    camera_matrix = list(struct.unpack("<9d", reader.read(72)))
    rectification = list(struct.unpack("<9d", reader.read(72)))
    projection = list(struct.unpack("<12d", reader.read(96)))
    return {
        "timestamp_ns": timestamp_ns,
        "width": width,
        "height": height,
        "distortion_model": distortion_model,
        "distortion_coefficients": distortion,
        "camera_matrix": camera_matrix,
        "rectification_matrix": rectification,
        "projection_matrix": projection,
    }


def read_first_camera_info(
    database: sqlite3.Connection,
    topics: dict[str, int],
    topic_names: tuple[str, ...],
) -> dict[str, object]:
    for topic_name in topic_names:
        topic_id = topics.get(topic_name)
        if topic_id is None:
            continue
        row = database.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
            (topic_id,),
        ).fetchone()
        if row is not None:
            return parse_camera_info(row[0])
    return {}


def read_first_string(
    database: sqlite3.Connection,
    topics: dict[str, int],
    topic_name: str,
) -> str | None:
    topic_id = topics.get(topic_name)
    if topic_id is None:
        return None
    row = database.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
        (topic_id,),
    ).fetchone()
    if row is None:
        return None
    reader = CdrReader(row[0])
    return reader.string()


def write_intrinsics(
    path: Path,
    color_info: dict[str, object],
    depth_info: dict[str, object],
    depth_scale: float | None,
    recording: Path,
) -> None:
    output = {
        "source_recording": recording.name,
        "depth_encoding": "16-bit PNG in recorded sensor units",
        "depth_scale_m_per_unit": depth_scale,
        "color": color_info,
        "depth": depth_info,
    }
    with path.open("w") as file:
        json.dump(output, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="ROS 2 SQLite bag (.db3)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--every", type=int, default=1, help="save every Nth matched pair")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--max-time-delta-ms",
        type=float,
        default=100.0,
        help="maximum color/depth timestamp difference (default: 100 ms)",
    )
    args = parser.parse_args()
    if not args.recording.is_file():
        parser.error(f"recording not found: {args.recording}")
    if args.every < 1:
        parser.error("--every must be at least 1")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.max_time_delta_ms < 0:
        parser.error("--max-time-delta-ms cannot be negative")
    return args


def iter_pairs(
    database: sqlite3.Connection,
    topics: dict[str, int],
    color_messages: list[BagMessage],
    depth_messages: list[BagMessage],
    max_delta_ns: int,
) -> Iterator[tuple[ImageMessage, ImageMessage]]:
    color_timestamps = [message.timestamp_ns for message in color_messages]
    for depth_message in depth_messages:
        color_message = nearest_message(
            color_messages,
            color_timestamps,
            depth_message.timestamp_ns,
        )
        if abs(color_message.timestamp_ns - depth_message.timestamp_ns) > max_delta_ns:
            continue
        yield (
            read_image(database, color_message, topics[COLOR_TOPIC], COLOR_TOPIC),
            read_image(database, depth_message, topics[DEPTH_TOPIC], DEPTH_TOPIC),
        )


def main() -> int:
    args = parse_args()
    rgb_dir = args.out_dir / "rgb"
    depth_dir = args.out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.recording) as database:
        topics = read_topic_ids(database)
        color_messages = read_messages(database, topics[COLOR_TOPIC])
        depth_messages = read_messages(database, topics[DEPTH_TOPIC])
        color_info = read_first_camera_info(database, topics, (COLOR_INFO_TOPIC,))
        depth_info = read_first_camera_info(
            database,
            topics,
            (DEPTH_INFO_TOPIC, DEPTH_INFO_TOPIC_FALLBACK),
        )
        depth_units = read_first_string(database, topics, DEPTH_UNITS_TOPIC)
        depth_scale = float(depth_units) if depth_units is not None else None
        write_intrinsics(
            args.out_dir / "intrinsics.json",
            color_info,
            depth_info,
            depth_scale,
            args.recording,
        )

        with (args.out_dir / "manifest.csv").open("w", newline="") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(
                [
                    "index",
                    "color_timestamp_ns",
                    "depth_timestamp_ns",
                    "timestamp_delta_ms",
                    "rgb_file",
                    "depth_file",
                    "color_message_id",
                    "depth_message_id",
                ]
            )
            matched = saved = 0
            for color, depth in iter_pairs(
                database,
                topics,
                color_messages,
                depth_messages,
                int(args.max_time_delta_ms * 1_000_000),
            ):
                if matched % args.every:
                    matched += 1
                    continue
                name = f"{saved:06d}.png"
                color_path = rgb_dir / name
                depth_path = depth_dir / name
                if not cv2.imwrite(str(color_path), color.image):
                    raise RuntimeError(f"could not write color frame: {color_path}")
                if not cv2.imwrite(str(depth_path), depth.image):
                    raise RuntimeError(f"could not write depth frame: {depth_path}")
                writer.writerow(
                    [
                        saved,
                        color.timestamp_ns,
                        depth.timestamp_ns,
                        (color.timestamp_ns - depth.timestamp_ns) / 1_000_000,
                        f"rgb/{name}",
                        f"depth/{name}",
                        color.message_id,
                        depth.message_id,
                    ]
                )
                matched += 1
                saved += 1
                if args.max_frames and saved >= args.max_frames:
                    break

    print(
        f"Matched {matched} color/depth pairs, saved {saved} pairs to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

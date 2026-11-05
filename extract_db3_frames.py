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

# constant topic names used by the ROS 2 RealSense driver
COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data/compressed"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data/compressedDepth"
COLOR_INFO_TOPIC = "/device_0/sensor_1/Color_0/image/camera_info"
DEPTH_INFO_TOPIC = "/device_0/sensor_0/Depth_0/image/camera_info"
DEPTH_INFO_TOPIC_FALLBACK = "/device_0/sensor_0/Depth_0/camera_info"
DEPTH_UNITS_TOPIC = "/device_0/sensor_0/option/Depth_Units/value"

class CdrReader:
    """Read the little-endian ROS 2 CDR fields used by these messages."""

    def __init__(self, data: bytes):
        # first 4 bytes are the CDR header, which is always 0x0001 followed by two reserved bytes.
        # reserved bytes are not used, but we still need to skip them.
        if data[:2] != b"\x00\x01":
            raise ValueError("unsupported ROS 2 CDR encoding")
        # CDR header is 4 bytes long, so we start reading from offset 4.
        self.data = data
        self.offset = 4

    # aligns the current offset to the specified boundary.
    def align(self, boundary: int) -> None:
        self.offset = (self.offset + boundary - 1) & ~(boundary - 1)

    # reads a specified number of bytes from the data starting at the current offset.
    def read(self, size: int) -> bytes:
        # calculate the end offset by adding the size to the current offset.
        end = self.offset + size
        # if the end offset exceeds the length of the data, raise a ValueError indicating that the ROS 2 CDR message is truncated.
        if end > len(self.data):
            raise ValueError("truncated ROS 2 CDR message")
        # extract the bytes from the current offset to the end offset and updates the current offset to the end offset.
        value = self.data[self.offset:end]
        self.offset = end
        return value

    # reads a scalar value of a specified format from the data.
    def scalar(self, fmt: str) -> int | float:
        # calculate the size of the scalar value using struct.calcsize and aligns the current offset to that size.
        size = struct.calcsize(fmt)
        self.align(size)
        return struct.unpack("<" + fmt, self.read(size))[0]

    # reads a CDR-encoded string from the data.
    def string(self) -> str:
        # reads the length of the string as a 32-bit unsigned integer using the scalar method.
        length = int(self.scalar("I"))
        # reads the string data of the specified length and checks if the last byte is a null terminator (0).
        raw = self.read(length)
        if not raw or raw[-1] != 0:
            raise ValueError("invalid ROS 2 CDR string")
        # decodes the string data (excluding the null terminator) from bytes to a UTF-8 string and returns it.
        return raw[:-1].decode("utf-8")

    # reads a CDR-encoded sequence of bytes from the data.
    def sequence(self, item_size: int) -> bytes:
        # reads the length of the sequence as a 32-bit unsigned integer using the scalar method.
        length = int(self.scalar("I"))
        self.align(item_size)
        return self.read(length * item_size)

    # reads a CDR-encoded sequence of scalar values from the data.
    def sequence_values(self, fmt: str) -> list[int | float]:
        # calculates the size of each item in the sequence using struct.calcsize and aligns the current offset to that size.
        item_size = struct.calcsize(fmt)
        length = int(self.scalar("I"))
        return list(struct.unpack("<" + fmt * length, self.read(item_size * length)))

# dataclass decorator is used to define two immutable data classes, BagMessage and ImageMessage, which represent messages in the ROS 2 bag file.
@dataclass(frozen=True)
class BagMessage:
    message_id: int
    timestamp_ns: int

@dataclass(frozen=True)
class ImageMessage:
    message_id: int
    timestamp_ns: int
    image: np.ndarray

# reads the header of a ROS 2 CDR message and returns the timestamp in nanoseconds.
def parse_header(reader: CdrReader) -> int:
    # reads the seconds and nanoseconds components of the timestamp from the CDR message using the scalar method.
    seconds = int(reader.scalar("i"))
    nanoseconds = int(reader.scalar("I"))
    # reads the frame_id string from the CDR message using the string method, but does not use it in the return value.
    reader.string()
    # combines the seconds and nanoseconds into a single timestamp in nanoseconds and returns it.
    return seconds * 1_000_000_000 + nanoseconds

# reads a ROS 2 compressed image message and returns the timestamp, format name, and compressed image data.
def parse_compressed_image(data: bytes) -> tuple[int, str, bytes]:
    # create a CdrReader instance to read the CDR-encoded data
    reader = CdrReader(data)
    # call parse_header to extract the timestamp.
    timestamp_ns = parse_header(reader)
    # read the format name string from the CDR message using the string method.
    format_name = reader.string()
    # read the compressed image data from the CDR message using the sequence method with an item size of 1 (byte).
    image_data = reader.sequence(1)
    return timestamp_ns, format_name, image_data

# decode a ROS compressed image, including compressedDepth PNG payloads.
def decode_image(data: bytes, topic: str) -> np.ndarray:
    """Decode a ROS compressed image, including compressedDepth PNG payloads."""
    # find the start of the PNG header in the data
    png_start = data.find(b"\x89PNG\r\n\x1a\n")
    # if the PNG header is not found, raise a ValueError indicating that the topic does not contain a PNG payload.
    if png_start < 0:
        raise ValueError(f"{topic} does not contain a PNG payload")

    # decode the PNG image data using OpenCV's imdecode function, which reads the image from a buffer and returns it as a NumPy array.
    image = cv2.imdecode(
        np.frombuffer(data[png_start:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    # if the image could not be decoded, raise a ValueError indicating that OpenCV could not decode an image from the specified topic.
    if image is None:
        raise ValueError(f"OpenCV could not decode an image from {topic}")
    return image

# read the topic IDs from the SQLite database and return a dictionary mapping topic names to their corresponding IDs.
def read_topic_ids(database: sqlite3.Connection) -> dict[str, int]:
    # execute a SQL query to select the id and name columns from the topics table in the database and fetch all rows.
    rows = database.execute("SELECT id, name FROM topics").fetchall()
    # create a dictionary mapping topic names to their corresponding IDs using a dictionary comprehension.
    topics = {name: topic_id for topic_id, name in rows}
    # check if the required topics (COLOR_TOPIC and DEPTH_TOPIC) are present in the topics dictionary. If any required topic is missing, raise a RuntimeError indicating which topic(s) are missing.
    required = (COLOR_TOPIC, DEPTH_TOPIC)
    missing = [topic for topic in required if topic not in topics]
    if missing:
        raise RuntimeError("recording is missing required topic(s): " + ", ".join(missing))
    return topics

# read the messages for a specific topic from the SQLite database and return a list of BagMessage instances.
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

# read a specific image message from the SQLite database and return an ImageMessage instance.
def read_image(
    database: sqlite3.Connection,
    message: BagMessage,
    topic_id: int,
    topic_name: str,
) -> ImageMessage:
    # execute a SQL query to select the data column from the messages table for the specified message ID and topic ID, and fetch one row.
    row = database.execute(
        "SELECT data FROM messages WHERE id = ? AND topic_id = ?",
        (message.message_id, topic_id),
    ).fetchone()
    # if no row is found, raise a RuntimeError indicating that the message has disappeared.
    if row is None:
        raise RuntimeError(f"message {message.message_id} disappeared from the bag")
    # parse the compressed image data from the row using the parse_compressed_image function, which returns the timestamp, format name, and compressed image data.
    timestamp_ns, _format_name, compressed_data = parse_compressed_image(row[0])
    # decode the compressed image data using the decode_image function, which returns the decoded image as a NumPy array.
    image = decode_image(compressed_data, topic_name)
    return ImageMessage(message.message_id, timestamp_ns, image)

# find the nearest message in a list of messages based on a given timestamp.
def nearest_message(
    messages: list[BagMessage],
    timestamps: list[int],
    timestamp_ns: int,
) -> BagMessage:
    # use the bisect module to find the position of the timestamp in the list of timestamps.
    position = bisect.bisect_left(timestamps, timestamp_ns)
    # create a list of candidate messages to consider for finding the nearest message
    candidates = []
    # if the position is within the bounds of the messages list, add the message at that position to the candidates list.
    if position < len(messages):
        candidates.append(messages[position])
    # if the position is greater than 0, add the message at position - 1 to the candidates list.
    if position:
        candidates.append(messages[position - 1])
    return min(candidates, key=lambda message: abs(message.timestamp_ns - timestamp_ns))

# parse the camera info message from the CDR-encoded data and return a dictionary containing the camera parameters.
def parse_camera_info(data: bytes) -> dict[str, object]:
    # create a CdrReader instance to read the CDR-encoded data
    reader = CdrReader(data)
    # call parse_header to extract the timestamp.
    timestamp_ns = parse_header(reader)
    # read the height and width of the camera image from the CDR message using the scalar method.
    height = int(reader.scalar("I"))
    width = int(reader.scalar("I"))
    # read the distortion model string from the CDR message using the string method.
    distortion_model = reader.string()
    # read the distortion coefficients, camera matrix, rectification matrix, and projection matrix from the CDR message using the sequence_values method with appropriate formats.
    distortion = reader.sequence_values("d")
    camera_matrix = list(struct.unpack("<9d", reader.read(72)))
    rectification = list(struct.unpack("<9d", reader.read(72)))
    projection = list(struct.unpack("<12d", reader.read(96)))
    # return a dictionary containing the parsed camera parameters.
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

# read the first camera info message for the specified topic names from the SQLite database and return a dictionary containing the camera parameters.
def read_first_camera_info(
    database: sqlite3.Connection,
    topics: dict[str, int],
    topic_names: tuple[str, ...],
) -> dict[str, object]:
    # iterate over the provided topic names to find the first available camera info message
    for topic_name in topic_names:
        # look up the topic ID in the topics dictionary
        topic_id = topics.get(topic_name)
        if topic_id is None:
            continue
        # execute a SQL query to select the data column from the messages table for the specified topic ID, ordered by timestamp, and fetch one row.
        row = database.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
            (topic_id,),
        ).fetchone()
        # if a row is found, parse the camera info message using the parse_camera_info function and return the resulting dictionary.
        if row is not None:
            return parse_camera_info(row[0])
    return {}

# read the first string message for the specified topic name from the SQLite database and return the string value.
def read_first_string(
    database: sqlite3.Connection,
    topics: dict[str, int],
    topic_name: str,
) -> str | None:
    # look up the topic ID in the topics dictionary
    topic_id = topics.get(topic_name)
    if topic_id is None:
        return None
    # execute a SQL query to select the data column from the messages table for the specified topic ID, ordered by timestamp, and fetch one row.
    row = database.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
        (topic_id,),
    ).fetchone()
    if row is None:
        return None
    # create a CdrReader instance to read the CDR-encoded data
    reader = CdrReader(row[0])
    return reader.string()

# write the camera intrinsics and depth metadata to a JSON file.
def write_intrinsics(
    path: Path,
    color_info: dict[str, object],
    depth_info: dict[str, object],
    depth_scale: float | None,
    recording: Path,
) -> None:
    # create a dictionary containing the camera intrinsics and depth metadata, including the source recording name, depth encoding, depth scale, color camera info, and depth camera info.
    output = {
        "source_recording": recording.name,
        "depth_encoding": "16-bit PNG in recorded sensor units",
        "depth_scale_m_per_unit": depth_scale,
        "color": color_info,
        "depth": depth_info,
    }
    # write the output dictionary to a JSON file at the specified path with an indentation of 2 spaces for readability.
    with path.open("w") as file:
        json.dump(output, file, indent=2)

# parse command-line arguments and validate them, returning the parsed arguments as an argparse.Namespace object.
def parse_args() -> argparse.Namespace:
    # create an ArgumentParser instance with a description of the script's functionality.
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

# iterate over pairs of color and depth messages, yielding synchronized ImageMessage instances.
def iter_pairs(
    database: sqlite3.Connection,
    topics: dict[str, int],
    color_messages: list[BagMessage],
    depth_messages: list[BagMessage],
    max_delta_ns: int,
) -> Iterator[tuple[ImageMessage, ImageMessage]]:
    # create a list of timestamps for the color messages to facilitate efficient searching for the nearest color message for each depth message.
    color_timestamps = [message.timestamp_ns for message in color_messages]
    # iterate over each depth message and find the nearest color message based on the timestamp.
    for depth_message in depth_messages:
        # find the nearest color message to the current depth message using the nearest_message function, which returns the closest BagMessage based on timestamp.
        color_message = nearest_message(
            color_messages,
            color_timestamps,
            depth_message.timestamp_ns,
        )
        # if the absolute difference between the timestamps of the color and depth messages exceeds the specified maximum delta, skip this pair and continue to the next depth message.
        if abs(color_message.timestamp_ns - depth_message.timestamp_ns) > max_delta_ns:
            continue
        # read the color and depth images from the database using the read_image function, which returns ImageMessage instances for each message.
        yield (
            read_image(database, color_message, topics[COLOR_TOPIC], COLOR_TOPIC),
            read_image(database, depth_message, topics[DEPTH_TOPIC], DEPTH_TOPIC),
        )

# main function that orchestrates the extraction of synchronized color and depth images from a ROS 2 SQLite bag file.
def main() -> int:
    # parse command-line arguments and create output directories for RGB and depth images
    args = parse_args()
    rgb_dir = args.out_dir / "rgb"
    depth_dir = args.out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # connect to the SQLite database and read the topic IDs, color messages, depth messages, camera info, and depth units
    with sqlite3.connect(args.recording) as database:
        # read the topic IDs from the database and store them in a dictionary
        topics = read_topic_ids(database)
        # read the color and depth messages from the database for the specified topics
        color_messages = read_messages(database, topics[COLOR_TOPIC])
        depth_messages = read_messages(database, topics[DEPTH_TOPIC])
        # read the camera info for color and depth topics
        color_info = read_first_camera_info(database, topics, (COLOR_INFO_TOPIC,))
        depth_info = read_first_camera_info(
            database,
            topics,
            (DEPTH_INFO_TOPIC, DEPTH_INFO_TOPIC_FALLBACK),
        )
        # read the depth units from the database and convert it to a float if available
        depth_units = read_first_string(database, topics, DEPTH_UNITS_TOPIC)
        depth_scale = float(depth_units) if depth_units is not None else None
        # write the camera intrinsics and depth metadata to a JSON file in the output directory
        write_intrinsics(
            args.out_dir / "intrinsics.json",
            color_info,
            depth_info,
            depth_scale,
            args.recording,
        )

        # create a CSV manifest file in the output directory to store information about the matched color and depth image pairs
        with (args.out_dir / "manifest.csv").open("w", newline="") as manifest_file:
            # create a CSV writer object to write rows to the manifest file
            writer = csv.writer(manifest_file)
            # write the header row to the manifest file, specifying the column names for the index, timestamps, file paths, and message IDs of the matched color and depth image pairs
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
            # write the matched color and depth image pairs to the manifest file
            matched = saved = 0
            # iterate over the synchronized color and depth image pairs using the iter_pairs function, which yields ImageMessage instances for each matched pair
            for color, depth in iter_pairs(
                database,
                topics,
                color_messages,
                depth_messages,
                int(args.max_time_delta_ms * 1_000_000),
            ):
                # if the current matched pair index is not a multiple of the specified --every argument, skip this pair and continue to the next one
                if matched % args.every:
                    matched += 1
                    continue
                # generate a filename for the saved color and depth images based on the current saved index, formatted as a zero-padded 6-digit number with a .png extension
                name = f"{saved:06d}.png"
                # save the color and depth images to the output directories using OpenCV's imwrite function, raising a RuntimeError if the images cannot be written
                color_path = rgb_dir / name
                depth_path = depth_dir / name
                if not cv2.imwrite(str(color_path), color.image):
                    raise RuntimeError(f"could not write color frame: {color_path}")
                if not cv2.imwrite(str(depth_path), depth.image):
                    raise RuntimeError(f"could not write depth frame: {depth_path}")
                # write a row to the manifest file containing the index, timestamps, file paths, and message IDs of the matched color and depth image pair
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
                #  increment the matched and saved counters
                matched += 1
                saved += 1
                # if a maximum number of frames is specified and the saved count has reached that limit, break out of the loop to stop processing further pairs
                if args.max_frames and saved >= args.max_frames:
                    break

    # print a summary of the number of matched color/depth pairs and the number of pairs saved to the output directory
    print(
        f"Matched {matched} color/depth pairs, saved {saved} pairs to {args.out_dir}"
    )
    return 0

# if the script is run as the main module, call the main function and exit with its return code
if __name__ == "__main__":
    raise SystemExit(main())

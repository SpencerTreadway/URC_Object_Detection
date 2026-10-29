import numpy as np
import pyrealsense2 as rs

# try starting the camera with different settings until one works
def try_start_camera():
    # list of candidate camera settings to try
    # RealSense needs both color and depth enabled together for the profile to include both streams
    candidates = [
        (640, 480, 30),
        (640, 480, 15),
    ]

    # try each candidate setting until one works
    last_error = None  # store the last error encountered while trying to start the camera

    # iterate over each candidate setting
    for width, height, fps in candidates:
        # attempt to start the camera with the current candidate setting
        pipeline = rs.pipeline()
        # create a new configuration for the pipeline
        config = rs.config()
        # enable both color and depth streams with the current candidate setting
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        try:
            # try to start the camera with the current configuration
            profile = pipeline.start(config)
            # if successful, return the pipeline, profile, and settings used
            return pipeline, profile, (width, height, fps)
        # if starting the pipeline failed, catch the exception and try the next candidate setting
        except RuntimeError as exc:
            # store the last error encountered while trying to start the camera
            last_error = exc
            # attempt to stop the pipeline to clean up resources
            try:
                # stop the pipeline to release any resources it may have acquired
                pipeline.stop()
            # continue to the next candidate setting
            except Exception:
                pass

    # if none of the candidate settings worked, raise an error with the last encountered exception
    raise RuntimeError(f"No supported combined color+depth stream was available for this RealSense device: {last_error}")


def camera_matrix_from_intrinsics(intrinsics) -> np.ndarray:
    '''Construct a camera matrix from RealSense stream intrinsics.'''

    # construct the camera matrix from the intrinsic parameters
    return np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def create_color_depth_alignment():
    '''Create an alignment object that maps depth pixels to the color frame.'''

    return rs.align(rs.stream.color)


def wait_for_color_and_depth_frames(pipeline, alignment=None, timeout_ms=None):
    '''Return color and depth frames, optionally aligned to the color frame.'''

    if timeout_ms is None:
        frames = pipeline.wait_for_frames()
    else:
        frames = pipeline.wait_for_frames(timeout_ms)
    if alignment is not None:
        frames = alignment.process(frames)
    return frames.get_color_frame(), frames.get_depth_frame()


def get_depth_distance(depth_frame, x: int, y: int) -> float:
    '''Return the distance in meters at one depth-frame pixel.'''

    return float(depth_frame.get_distance(x, y))


def get_bounding_box_distance(
    depth_frame,
    bounding_box: tuple[int, int, int, int],
) -> float:
    '''Return the median distance in meters from the valid pixels in a bounding box.'''

    left, top, right, bottom = bounding_box
    distances = []
    for y in range(top, bottom):
        for x in range(left, right):
            distance = get_depth_distance(depth_frame, x, y)
            if distance > 0:
                distances.append(distance)
    if not distances:
        raise RuntimeError("No valid depth data was found in the bounding box")
    return float(np.median(distances))

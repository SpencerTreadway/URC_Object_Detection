import cv2
import numpy as np
import sys

def open_camera(source: int | str) -> cv2.VideoCapture:
    '''Open a camera index or a video stream URL.'''

    # determine the appropriate backend for the camera based on the operating system
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    # create a VideoCapture object with the specified source and backend
    camera = cv2.VideoCapture(source, backend)

    # check if the camera was successfully opened; if not, release it and raise a RuntimeError
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera source: {source}. Try another camera index." # D415 color stream may be distinct from its depth/IR streams.
        )
    
    # Match the 640x480 resolution used to generate color_camera_calibration.npz.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # return the VideoCapture object
    return camera

def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''Load camera_matrix and distortion coefficients from an NPZ file.'''

    # load the calibration data from the specified NPZ file
    calibration = np.load(path)
    try:
        # return the camera_matrix and distortion coefficients from the calibration data
        return calibration["K"], calibration["dist"]
    except KeyError as error:
        # raise a ValueError if the required arrays are not found in the calibration data
        raise ValueError(
            f"{path} must contain camera_matrix and dist_coeffs arrays"
        ) from error

def scale_camera_matrix(
    camera_matrix: np.ndarray,
    calibration_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> np.ndarray:
    '''Scale calibration intrinsics when the capture resolution differs.'''

    # unpack the calibration and frame sizes into width and height
    calibration_width, calibration_height = calibration_size
    frame_width, frame_height = frame_size
    # calculate the scaling factors for the x and y dimensions based on the frame and calibration sizes
    scale_x = frame_width / calibration_width
    scale_y = frame_height / calibration_height
    # create a copy of the camera matrix to avoid modifying the original one
    scaled_matrix = camera_matrix.astype(np.float64, copy=True)
    # scale the focal lengths and principal point coordinates by the calculated factors
    scaled_matrix[0, 0] *= scale_x
    scaled_matrix[1, 1] *= scale_y
    scaled_matrix[0, 2] *= scale_x
    scaled_matrix[1, 2] *= scale_y
    # return the scaled camera matrix
    return scaled_matrix

def approximate_camera_matrix(frame) -> np.ndarray:
    '''Create rough intrinsics for testing when no calibration is available.'''

    # get the height and width of the frame
    height, width = frame.shape[:2]
    # use the width as the focal length
    focal_length = width
    # create and return a rough camera matrix with the focal length and principal point at the center of the frame
    return np.array(
        [
            [focal_length, 0, width / 2], # focal length in x direction, 0 skew, principal point x
            [0, focal_length, height / 2], # 0 skew, focal length in y direction, principal point y
            [0, 0, 1], # homogeneous coordinate
        ],
        dtype=np.float32,
    )
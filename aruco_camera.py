import argparse
from dataclasses import dataclass
import os
import shutil
import cv2
import numpy as np
import sys
from camera_init import *

@dataclass
class DetectorConfig:
    '''Settings that control which ArUco markers are detected.'''

    dictionary_name: str = "DICT_4X4_50"
    camera_source: int | str = 0
    window_name: str = "ArUco camera"
    marker_size_m: float = 0.02
    calibration_file: str | None = "color_camera_calibration.npz"
    calibration_width: int = 640
    calibration_height: int = 480

def create_detector(dictionary_name: str):
    '''Create an ArUco detector from a predefined OpenCV dictionary.'''

    try:
        # get dictionary ID from the cv2.aruco module using the provided dictionary name
        dictionary_id = getattr(cv2.aruco, dictionary_name)
    except AttributeError as error:
        # raise a ValueError if the dictionary name is not found in cv2.aruco
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}") from error

    # create the dictionary and detector parameters
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()

    # return the ArUco detector if it exists, otherwise return the dictionary and parameters
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters

def detect_markers(frame, detector):
    '''Return marker corners, IDs, and rejected candidates for one frame.'''

    # if the detector is a tuple (dictionary, parameters), use cv2.aruco.detectMarkers
    if isinstance(detector, tuple):
        # unpack the dictionary and parameters from the detector tuple
        dictionary, parameters = detector
        # use cv2.aruco.detectMarkers to detect markers in the frame
        return cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    # if the detector is an ArUcoDetector object, use its detectMarkers methodod
    return detector.detectMarkers(frame)

def draw_marker_corners(frame: np.ndarray, corners) -> None:
    '''Draw a visible point at each corner of every detected marker.'''

    # iterate through the detected marker corners
    for marker_corners in corners:
        # reshape the marker corners to a 4x2 array and draw a circle at each corner on the frame
        for x, y in marker_corners.reshape(4, 2):
            cv2.circle(
                frame,
                (int(round(x)), int(round(y))),
                5,
                (0, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

def estimate_marker_poses(corners, marker_size_m: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    '''Estimate each marker's rotation and translation from the camera.'''

    # calculate half the size of the marker in meters
    half_size = marker_size_m / 2
    # define the 3D coordinates of the marker corners in the marker's local coordinate system
    object_points = np.array(
        [
            [-half_size, half_size, 0], # top-left corner
            [half_size, half_size, 0], # top-right corner
            [half_size, -half_size, 0], # bottom-right corner
            [-half_size, -half_size, 0], # bottom-left corner
        ],
        dtype=np.float32,
    )

    # initialize lists to hold the rotation and translation vectors for each detected marker
    rvecs = []
    tvecs = []
    for marker_corners in corners:
        # reshape the marker corners to a 4x2 array and convert to float32
        image_points = marker_corners.reshape(4, 2).astype(np.float32)
        # use cv2.solvePnP to estimate the rotation and translation vectors for the marker
        solved, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        # check if the pose estimation was successful; if not, raise a RuntimeError
        if not solved:
            raise RuntimeError("OpenCV could not estimate the marker pose")
        # append the estimated rotation and translation vectors to the respective lists
        rvecs.append(rvec)
        tvecs.append(tvec)
    # convert the lists of rotation and translation vectors to numpy arrays and return them
    return np.asarray(rvecs), np.asarray(tvecs)

def describe_pose(translation) -> tuple[float, float, float, float, float, float]:
    '''Return x, y, z, distance, horizontal, and vertical bearing in degrees.'''
    # reshape the translation vector to a 3-element array and unpack the x, y, z coordinates
    x, y, z = translation.reshape(3)
    # calculate the Euclidean distance from the camera to the marker
    distance = float(np.linalg.norm([x, y, z]))
    # calculate the horizontal bearing in degrees using arctangent of x and z
    horizontal_bearing = float(np.degrees(np.arctan2(x, z)))
    # calculate the vertical bearing in degrees using arctangent of -y and z
    vertical_bearing = float(np.degrees(np.arctan2(-y, z)))
    # return the x, y, z coordinates, distance, horizontal bearing, and vertical bearing as floats
    return float(x), float(y), float(z), distance, horizontal_bearing, vertical_bearing

def clear_console() -> None:
    '''Clear the terminal without failing when output is redirected.'''
    if not sys.stdout.isatty():
        return
    command = shutil.which("clear") if os.name != "nt" else shutil.which("cls")
    if command:
        os.system(command)

def print_detection_table(detections: list[tuple[int, float, float, float, float, float, float]]) -> None:
    '''Print the current marker poses in a compact, stable table.'''

    clear_console()
    print("ArUco detections (camera coordinates; x right, y down, z forward)")
    print("ID    X (m)    Y (m)    Z (m)   Dist (m)  Bearing  Elevation")
    print("----  -------  -------  -------  --------  -------  ---------")
    if not detections:
        print("none")
        return
    for marker_id, x, y, z, distance, bearing, elevation in sorted(detections):
        print(
            f"{marker_id:>4}  {x:>7.3f}  {y:>7.3f}  {z:>7.3f}  "
            f"{distance:>8.3f}  {bearing:>6.1f} deg  {elevation:>8.1f} deg"
        )

def run(config: DetectorConfig) -> None:
    '''Capture frames until the user presses q or Escape.'''

    # open the camera using the specified source from the configuration
    camera = open_camera(config.camera_source)
    # create the ArUco detector using the specified dictionary name from the configuration
    detector = create_detector(config.dictionary_name)
    # initialize the camera matrix and distortion coefficients to None; they will be loaded or approximated later
    camera_matrix = None
    dist_coeffs = None

    try:
        # continuously capture frames from the camera and process them until the user presses 'q' or Escape
        while True:
            # read a frame from the camera
            success, frame = camera.read()
            # if unsuccessful, print an error message and break the loop
            if not success:
                print("Could not read a frame from the camera.")
                break

            # convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect ArUco markers in the grayscale frame using the detector
            corners, ids, _rejected = detect_markers(gray_frame, detector)

            # initialize an empty list to hold the detected marker poses
            detections = []
            # if any markers were detected, process each detected marker
            if ids is not None:
                # draw the detected markers on the original frame
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                draw_marker_corners(frame, corners)
                # if the camera matrix has not been loaded or approximated yet, load it from a calibration file or approximate it
                if camera_matrix is None:
                    # if a calibration file is specified in the configuration, load the camera matrix and distortion coefficients from that file
                    if config.calibration_file:
                        camera_matrix, dist_coeffs = load_calibration(config.calibration_file)
                        camera_matrix = scale_camera_matrix(
                            camera_matrix,
                            (config.calibration_width, config.calibration_height),
                            (frame.shape[1], frame.shape[0]),
                        )
                    # if no calibration file is specified, approximate the camera matrix and set distortion coefficients to zero, while printing a warning message
                    else:
                        camera_matrix = approximate_camera_matrix(frame)
                        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                        print(
                            "Warning: using approximate camera calibration; "
                            "distance is not accurate."
                        )
                    print(
                        f"Using marker size {config.marker_size_m:.3f} m with "
                        f"{frame.shape[1]}x{frame.shape[0]} frames."
                    )

                # estimate the rotation and translation vectors for each detected marker using the detected corners, marker size, camera matrix, and distortion coefficients
                rvecs, tvecs = estimate_marker_poses(
                    corners,
                    config.marker_size_m,
                    camera_matrix,
                    dist_coeffs,
                )

                # for each detected marker, draw its coordinate axes on the frame and describe its pose in terms of position and orientation
                for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
                    # draw the coordinate axes of the detected marker on the original frame using the estimated rotation and translation vectors, camera matrix, distortion coefficients, and half the marker size
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        config.marker_size_m / 2,
                    )
                    # describe the pose of the detected marker in terms of x, y, z coordinates, distance from the camera, horizontal bearing, and vertical elevation
                    x, y, z, distance, bearing, elevation = describe_pose(tvec)
                    # append the marker ID and its pose information to the detections list as a tuple
                    detections.append(
                        (int(marker_id), x, y, z, distance, bearing, elevation)
                    )

            # print the current marker poses in a compact, stable table format
            print_detection_table(detections)

            # display the original frame with the detected markers and their coordinate axes in a window with the specified name from the configuration
            cv2.imshow(config.window_name, frame)
            # wait for a key press for 1 millisecond and check if the pressed key is 'q' or Escape; if so, break the loop to exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        # release the camera and destroy all OpenCV windows when exiting the loop
        camera.release()
        cv2.destroyAllWindows()

def parse_source(value: str) -> int | str:
    '''Treat numeric command-line sources as camera indices.'''

    try:
        return int(value)
    except ValueError:
        return value

def main() -> None:
    # create an argument parser to handle command-line arguments for the script
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or stream URL (default: 0)",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV predefined ArUco dictionary name",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.02,
        metavar="METERS",
        help="Printed side length of the marker in meters (default: 0.02)",
    )
    parser.add_argument(
        "--calibration",
        metavar="FILE.npz",
        default="color_camera_calibration.npz",
        help="NPZ calibration file (default: color_camera_calibration.npz)",
    )
    args = parser.parse_args()

    # create a DetectorConfig object using the parsed command-line arguments and run the main detection loop
    config = DetectorConfig(
        dictionary_name=args.dictionary,
        camera_source=parse_source(args.source),
        marker_size_m=args.marker_size,
        calibration_file=args.calibration,
    )
    # run the main detection loop with the specified configuration
    run(config)

# if the script is being run directly (not imported as a module), call the main function to start the program
if __name__ == "__main__":
    main()
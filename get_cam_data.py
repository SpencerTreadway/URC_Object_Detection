import numpy as np
import pyrealsense2 as rs
from realsense_init import camera_matrix_from_intrinsics, try_start_camera

# main function to start the camera and print its intrinsic parameters
def main() -> int:
    # check if any RealSense devices are connected
    device_count = len(rs.context().query_devices())
    if device_count == 0:
        print("No RealSense camera detected. Please connect a supported Intel RealSense device.")
        return 1

    pipeline = None
    # attempt to start the camera with the available settings
    try:
        # try to start the camera and get the pipeline, profile, and settings
        pipeline, profile, settings = try_start_camera()

        # extract the color stream profile and its intrinsic parameters
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_stream.get_intrinsics()

        # construct the color camera matrix from the intrinsic parameters
        color_camera_matrix = camera_matrix_from_intrinsics(color_intrinsics)

        # extract the depth stream profile and its intrinsic parameters
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_stream.get_intrinsics()

        # construct the depth camera matrix from the intrinsic parameters
        depth_camera_matrix = camera_matrix_from_intrinsics(depth_intrinsics)

        # save both calibration datasets as separate .npz files
        np.savez("color_camera_calibration.npz", K=color_camera_matrix, dist=color_intrinsics.coeffs)
        np.savez("depth_camera_calibration.npz", K=depth_camera_matrix, dist=depth_intrinsics.coeffs)

        # print the camera settings and intrinsic parameters
        print(f"Using RealSense camera settings: {settings}")
        print("Color Intrinsic Camera Matrix:\n", color_camera_matrix)
        print("Color Distortion Coefficients:", color_intrinsics.coeffs)
        print("Depth Intrinsic Camera Matrix:\n", depth_camera_matrix)
        print("Depth Distortion Coefficients:", depth_intrinsics.coeffs)
        print("Saved calibration files: color_camera_calibration.npz and depth_camera_calibration.npz")
        return 0
    # handle any runtime errors that occurred while trying to start the camera
    except RuntimeError as exc:
        print(f"Unable to start the RealSense pipeline: {exc}")
        return 1
    finally:
        # attempt to stop the pipeline to release resources, regardless of whether an error occurred
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
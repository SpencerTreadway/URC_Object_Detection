"""Rigid K552 localization; no camera capture, ROS, or motor commands.

Coordinates: keyboard origin at top-left, x right, y down, z into the board.
Camera coordinates: x right, y down, z forward. All physical values are meters.
DEFAULT GEOMETRY IS A TEMPLATE: measure marker offsets and key centers first.
Requires numpy and opencv-contrib-python (cv2.aruco).
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, Mapping, Optional, Sequence, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

# Named types document array contents and distinguish pixels from metric points.
Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
FloatArray = NDArray[np.float64]
Pose = tuple[FloatArray, FloatArray]  # OpenCV rotation vector, translation vector
Selection = Literal["consensus", "single_tag", "fallback", "failed"]


class KeyLocation(TypedDict):
    """One key center expressed in the input image and camera coordinate frame."""

    pixel_xy: Point2D
    camera_xyz_m: Point3D


KEYBOARD_WIDTH_M = 0.354
KEYBOARD_HEIGHT_M = 0.123  # in-plane depth, not the 37 mm chassis thickness
KEYBOARD_THICKNESS_M = 0.037
TAG_SIZE_M = 0.02  # black square side, excluding surrounding white margin
TAG_DICTIONARY = "DICT_4X4_50"
# PLACEHOLDERS, ordered by fallback priority. Replace with actual assignments.
TAG_IDS = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}
AGREEMENT_PROPORTION = 0.05


def _template_keys() -> dict[str, Point3D]:
    """Build approximate ANSI 87-key centers from nominal key widths.

    Widths are multiples of a nominal key pitch. These values only provide a
    useful template; they do not account for measured K552 margins or key heights.
    """
    keys: dict[str, Point3D] = {}
    pitch = 0.019
    left = 0.006

    def row(names: Sequence[str], widths: Sequence[float], y: float) -> None:
        """Add one row, placing each center halfway across its key width."""
        x = left
        for name, width in zip(names, widths):
            keys[name] = (x + width * pitch / 2, y, 0.0)
            x += width * pitch

    row(["`", *"1234567890", "-", "=", "BACKSPACE"], [1] * 13 + [2], 0.035)
    row(["TAB", *"QWERTYUIOP", "[", "]", "\\"], [1.5] + [1] * 12 + [1.5], 0.054)
    row(
        ["CAPSLOCK", *"ASDFGHJKL", ";", "'", "ENTER"], [1.75] + [1] * 11 + [2.25], 0.073
    )
    row(
        ["LSHIFT", *"ZXCVBNM", ",", ".", "/", "RSHIFT"],
        [2.25] + [1] * 10 + [2.75],
        0.092,
    )
    row(
        ["LCTRL", "LWIN", "LALT", "SPACE", "RALT", "RWIN", "MENU", "RCTRL"],
        [1.25] * 3 + [6.25] + [1.25] * 4,
        0.111,
    )
    keys["ESC"] = (left + pitch / 2, 0.010, 0.0)
    for i in range(12):
        keys[f"F{i+1}"] = (left + (2.5 + i + (i // 4) * 0.5) * pitch, 0.010, 0.0)
    for names, y in [
        (["PRINTSCREEN", "SCROLLLOCK", "PAUSE"], 0.010),
        (["INSERT", "HOME", "PAGEUP"], 0.035),
        (["DELETE", "END", "PAGEDOWN"], 0.054),
        (["LEFT", "DOWN", "RIGHT"], 0.111),
    ]:
        for i, name in enumerate(names):
            keys[name] = (0.305 + i * pitch, y, 0.0)
    keys["UP"] = (0.324, 0.092, 0.0)
    return keys


KEY_LOCATIONS_M = _template_keys()


@dataclass(frozen=True)
class TagSpec:
    """A marker's measured geometry relative to the keyboard reference frame.

    Markers are parallel to the keyboard plane. Positive rotation is clockwise
    when viewing the keyboard face; center_m may include a depth offset.
    """

    tag_id: int
    center_m: Point3D
    size_m: float = TAG_SIZE_M
    rotation_deg: float = 0.0  # clockwise in keyboard plane

    def corners(self) -> FloatArray:
        """Return four keyboard-frame XYZ corners in decoded ArUco order.

        Start with TL, TR, BR, BL in the marker's unrotated frame, then apply
        its configured in-plane rotation and translate to its mounting position.
        """
        half = self.size_m / 2
        points = np.array(
            [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
            dtype=np.float64,
        )
        angle = np.deg2rad(self.rotation_deg)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        return points @ rotation.T + np.asarray(self.center_m)


def _template_tags() -> tuple[TagSpec, ...]:
    """Build placeholder marker geometry in top-left-first fallback order.

    Centers coincide with chassis corners; replace these with measured offsets.
    """
    return tuple(
        TagSpec(TAG_IDS[name], (x, y, 0.0))
        for name, x, y in [
            ("top_left", 0, 0),
            ("top_right", KEYBOARD_WIDTH_M, 0),
            ("bottom_right", KEYBOARD_WIDTH_M, KEYBOARD_HEIGHT_M),
            ("bottom_left", 0, KEYBOARD_HEIGHT_M),
        ]
    )


@dataclass
class KeyboardConfig:
    """Physical model and tag agreement policy, independent of the camera.

    The tags tuple defines fallback priority. agreement_proportion is a fraction
    of the keyboard diagonal, not a fraction of the camera-to-keyboard distance.
    Leave geometry_calibrated false until all placeholder geometry is replaced.
    """

    tags: tuple[TagSpec, ...] = field(default_factory=_template_tags)
    key_locations_m: Mapping[str, Point3D] = field(
        default_factory=lambda: dict(KEY_LOCATIONS_M)
    )
    width_m: float = KEYBOARD_WIDTH_M
    height_m: float = KEYBOARD_HEIGHT_M
    agreement_proportion: float = AGREEMENT_PROPORTION
    geometry_calibrated: bool = False


@dataclass
class DetectionResult:
    """Locations and diagnostics for one image; failed results have no locations.

    Visible IDs count configured tags even if pose estimation failed. Used IDs
    identify the selected pose group; rejected IDs are the other visible tags.
    ready_for_control only checks calibrated geometry and multi-tag agreement;
    downstream code must still handle robot calibration and motion constraints.
    """

    success: bool
    locations: dict[str, KeyLocation]
    visible_tag_ids: tuple[int, ...]
    used_tag_ids: tuple[int, ...]
    rejected_tag_ids: tuple[int, ...]
    total_detected_tags: int  # also includes unrelated dictionary tags
    selection: Selection
    geometry_calibrated: bool
    ready_for_control: bool
    keyboard_to_camera: Optional[FloatArray] = None
    reprojection_error_px: Optional[float] = None
    message: str = ""

    @property
    def num_tags_visible(self) -> int:
        """Number of configured tags detected in this image."""
        return len(self.visible_tag_ids)


class KeyboardDetector:
    """Localize a rigid keyboard from individual marker poses and their agreement.

    Construct once and reuse across images. Each result is computed independently;
    no tracking, temporal smoothing, or previous-frame fallback is performed.
    """

    def __init__(
        self,
        camera_matrix: FloatArray,
        dist_coeffs: FloatArray,
        config: Optional[KeyboardConfig] = None,
    ) -> None:
        """Intrinsics must match input image resolution and distortion model.

        Use camera_init.load_calibration and scale_camera_matrix if necessary.
        Configure actual tag IDs, centers, rotations, and key centers before
        setting geometry_calibrated=True. First configured tag is top-left.
        """
        self.config = config or KeyboardConfig()
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

        # Bad geometry or intrinsics should fail at setup, before processing frames.
        if (
            self.camera_matrix.shape != (3, 3)
            or not np.isfinite(self.camera_matrix).all()
            or self.camera_matrix[0, 0] <= 0
            or self.camera_matrix[1, 1] <= 0
            or not np.isfinite(self.dist_coeffs).all()
        ):
            raise ValueError("Invalid camera calibration")
        cfg = self.config
        if (
            not cfg.tags
            or len(cfg.tags) > 4
            or len({t.tag_id for t in cfg.tags}) != len(cfg.tags)
        ):
            raise ValueError("Configure one to four unique tags in fallback order")
        if not (
            0 < cfg.agreement_proportion < 1 and cfg.width_m > 0 and cfg.height_m > 0
        ):
            raise ValueError("Invalid dimensions or agreement proportion")
        for tag in cfg.tags:
            if (
                not 0 <= tag.tag_id < 50
                or not np.isfinite(tag.size_m)
                or tag.size_m <= 0
                or np.asarray(tag.center_m).shape != (3,)
                or not np.isfinite(tag.corners()).all()
            ):
                raise ValueError("Invalid tag geometry or DICT_4X4_50 ID")
        if not cfg.key_locations_m:
            raise ValueError("Key map must not be empty")
        for point in cfg.key_locations_m.values():
            if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Key centers must be finite XYZ coordinates in meters")

        # Support both OpenCV's class-based detector and its older function API.
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, TAG_DICTIONARY)
        )
        params = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (
            cv2.aruco.ArucoDetector(dictionary, params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else (dictionary, params)
        )

        # Evaluate pose agreement across the whole keyboard, not just a tag center.
        self.probes = np.array(
            [
                [0, 0, 0],
                [cfg.width_m, 0, 0],
                [cfg.width_m, cfg.height_m, 0],
                [0, cfg.height_m, 0],
            ]
        )

    @staticmethod
    def _transform(points: FloatArray, pose: Pose) -> FloatArray:
        """Apply a keyboard-to-camera rigid pose to an N-by-3 point array."""
        rvec, tvec = pose
        return np.asarray(points) @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3)

    def _select_consensus(
        self, poses: Mapping[int, Pose]
    ) -> tuple[tuple[int, ...], Selection]:
        """Largest mutually agreeing set; ties go to smallest disagreement.

        Pair distance = maximum 3D displacement of corresponding keyboard
        corners / physical keyboard diagonal. This checks rotation AND position.
        'Closest pair' means closest pose predictions, not adjacent physical tags.
        """
        ids = [tag.tag_id for tag in self.config.tags if tag.tag_id in poses]
        transformed = {i: self._transform(self.probes, poses[i]) for i in ids}
        diagonal = np.hypot(self.config.width_m, self.config.height_m)
        distances = {
            (a, b): float(
                np.linalg.norm(transformed[a] - transformed[b], axis=1).max() / diagonal
            )
            for a, b in combinations(ids, 2)
        }

        # With at most four markers, enumerate every group. Every pair must agree:
        # A agreeing with B and B with C does not imply A agrees with C.
        for size in range(len(ids), 1, -1):
            groups = []
            for group in combinations(ids, size):
                errors = [distances[pair] for pair in combinations(group, 2)]
                if max(errors) <= self.config.agreement_proportion:
                    groups.append((float(np.mean(errors)), group))
            if groups:
                return min(groups, key=lambda item: item[0])[1], "consensus"

        # TL when available, otherwise next visible tag in configured priority.
        return tuple(ids[:1]), "single_tag" if len(ids) == 1 else "fallback"

    def detect(
        self, image: NDArray[np.uint8], key: Optional[str] = None
    ) -> DetectionResult:
        """Process a uint8 BGR/grayscale image; key=None returns the full map.

        A specific key (case-insensitive) returns a one-entry locations map.
        Missing/ambiguous detections return success=False, never stale locations.

        Raises:
            KeyError: The requested key is absent from the configured layout.
            ValueError: The image is empty or is not uint8 grayscale/BGR.
        """
        if key is not None:
            key = key.upper()
            if key not in self.config.key_locations_m:
                raise KeyError(f"Unknown key: {key}")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.size == 0
            or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))
        ):
            raise ValueError("Expected a nonempty uint8 grayscale or BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if isinstance(self.detector, tuple):
            dictionary, params = self.detector
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=params
            )
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)
        return self._from_markers(corners, ids, key)

    def _from_markers(
        self,
        corners: Sequence[FloatArray],
        ids: Optional[NDArray[np.integer]],
        key: Optional[str] = None,
    ) -> DetectionResult:
        """Estimate, select, and combine poses from decoded marker observations.

        This separate geometry stage also allows tests to supply exact synthetic
        corner observations without running the image detector.
        """
        detected_ids = (
            [] if ids is None else [int(i) for i in np.asarray(ids).reshape(-1)]
        )
        visible = tuple(t.tag_id for t in self.config.tags if t.tag_id in detected_ids)

        def failure(message: str) -> DetectionResult:
            """Keep detection diagnostics while discarding unusable key locations."""
            return DetectionResult(
                False,
                {},
                visible,
                (),
                visible,
                len(detected_ids),
                "failed",
                self.config.geometry_calibrated,
                False,
                message=message,
            )

        if any(detected_ids.count(i) > 1 for i in visible):
            return failure("Duplicate configured tag ID; correspondence is ambiguous")
        observed = {
            i: np.asarray(c, dtype=np.float64).reshape(4, 2)
            for i, c in zip(detected_ids, corners)
        }
        poses: dict[int, Pose] = {}

        # Each marker independently predicts the entire keyboard pose because its
        # model corners are expressed in keyboard coordinates, not tag coordinates.
        for tag in self.config.tags:
            if tag.tag_id not in observed:
                continue
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    tag.corners(),
                    observed[tag.tag_id],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok and np.isfinite(rvec).all() and np.isfinite(tvec).all():
                    pose = (rvec, tvec)
                    if (self._transform(self.probes, pose)[:, 2] > 0).all():
                        poses[tag.tag_id] = pose
            except cv2.error:
                continue
        if not poses:
            return failure("No configured tag has a usable pose")

        selected, selection = self._select_consensus(poses)
        object_points = np.concatenate(
            [t.corners() for i in selected for t in self.config.tags if t.tag_id == i]
        )
        image_points = np.concatenate([observed[i] for i in selected])
        rvec, tvec = (v.copy() for v in poses[selected[0]])

        if len(selected) > 1:
            # Fit one rigid pose to all selected tag corners, rather than average Euler angles.
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                return failure("Combined pose estimation failed")
            if not ok:
                return failure("Combined pose estimation failed")
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return failure("Nonfinite keyboard pose")

        # Transform the same fixed key map into metric camera coordinates and
        # distorted image pixels. The physical key spacing is never stretched.
        names = [key] if key is not None else list(self.config.key_locations_m)
        points = np.array(
            [self.config.key_locations_m[name] for name in names], dtype=np.float64
        )
        camera_points = self._transform(points, (rvec, tvec))
        if (camera_points[:, 2] <= 0).any():
            return failure("Predicted keys lie behind camera")

        pixels = cv2.projectPoints(
            points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )[0].reshape(-1, 2)
        reprojected = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((reprojected - image_points) ** 2, axis=1))))

        # Homogeneous transform lets downstream code compose camera/robot frames.
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec.reshape(3)
        locations: dict[str, KeyLocation] = {
            name: {
                "pixel_xy": (float(pixel[0]), float(pixel[1])),
                "camera_xyz_m": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            }
            for name, pixel, xyz in zip(names, pixels, camera_points)
        }
        return DetectionResult(
            True,
            locations,
            visible,
            selected,
            tuple(i for i in visible if i not in selected),
            len(detected_ids),
            selection,
            self.config.geometry_calibrated,
            self.config.geometry_calibrated and len(selected) >= 2,
            transform,
            rms,
            (
                ""
                if self.config.geometry_calibrated
                else "Template geometry: not calibrated for arm control"
            ),
        )

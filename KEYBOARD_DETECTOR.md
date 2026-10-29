# Keyboard detector

## Current algorithm

The detector uses a fixed, rigid model of the Redragon K552 keyboard: its
354 x 123 mm chassis, an approximate 87-key center map, and four configurable
markers. Marker IDs and mounting positions are placeholders. Keyboard coordinates
start at the top-left, with x right, y down, and z into the keyboard.

For each image:

1. Detect DICT_4X4_50 markers and refine their image corners.
2. Use camera calibration and each marker's known geometry to independently
   estimate the keyboard pose with OpenCV PnP. Discard failed estimates and
   estimates that place the keyboard behind the camera.
3. Compare each pair by transforming the four chassis corners into camera
   coordinates. Disagreement is the largest distance between corresponding
   corners divided by the keyboard diagonal. The default threshold is 5%, about
   18.7 mm. This checks both position and orientation.
4. Select the largest group in which every pair agrees. Among equally large
   groups, choose the smallest mean disagreement. Three agreeing tags beat a
   pair; two separate agreeing pairs are resolved by their pose agreement,
   not physical proximity. Exact ties follow configured tag order.
5. If no pair agrees, use top-left when it has a usable pose. Otherwise use the
   first usable tag in configured fallback order.
6. Refit one rigid pose using all selected markers' corners. Transform the
   requested key center or full map into camera-relative meters and image pixels.
   The physical key map is never stretched or resized.

Results include visible, used, and rejected tag IDs; number of configured tags
visible; total detected tags including unrelated IDs; selection mode; a 4x4
keyboard-to-camera transform; and selected-corner reprojection error. Camera axes
are right/down/forward. Pixel coordinates include lens distortion and may lie
outside the image.

No usable tags or duplicate configured IDs produce a failed result with an empty
map. Each image is processed independently. `ready_for_control` requires measured
geometry marked calibrated and at least two agreeing tags. It is a preliminary
check, not validation of an arm movement or registered keypress.

## Remaining steps

1. Replace approximate key centers with K552 measurements, including their depth
   relative to the marker reference plane.
2. Assign actual marker IDs and measure centers, black-square side lengths, and
   in-plane rotations. Tags must be parallel to the keyboard plane; depth offsets
   are supported. Configure top-left first for fallback priority.
3. Verify camera intrinsics and distortion match the image stream and resolution.
   Resize intrinsics for resized images; adjust the principal point for crops.
4. Measure accuracy on real images across expected distances and angles. Tune the
   agreement threshold and test occlusion and misplaced tags before marking
   geometry calibrated. Synthetic tests currently cover selection rules,
   coordinate recovery, failure cases, and detection on rendered marker images.
5. Add any required temporal stability and quality checks before arm motion.
6. Connect to ROS, calibrate the camera-to-arm transform, and implement approach,
   press, retract, verification, and autonomous attempt coordination.

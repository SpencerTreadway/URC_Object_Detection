"""Synthetic geometry and real ArUco image tests; no camera required."""
import unittest

import cv2
import numpy as np

from keyboard_detector import KeyboardConfig, KeyboardDetector, KEY_LOCATIONS_M


class KeyboardDetectorTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[1000., 0, 400], [0, 1000., 300], [0, 0, 1]])
        self.detector = KeyboardDetector(self.K, np.zeros(5))
        self.rvec = np.array([0.15, -0.2, 0.03])
        self.tvec = np.array([-0.15, -0.05, 0.8])

    def poses(self, offsets):
        return {i: (self.rvec.copy(), self.tvec + [offset, 0, 0])
                for i, offset in enumerate(offsets)}

    def markers(self, offsets):
        corners, ids = [], []
        for tag, offset in zip(self.detector.config.tags, offsets):
            corners.append(cv2.projectPoints(tag.corners(), self.rvec,
                           self.tvec + [offset, 0, 0], self.K, np.zeros(5))[0])
            ids.append([tag.tag_id])
        return corners, np.array(ids)

    def test_consensus_rules(self):
        cases = [([0, .001, .002, .003], (0, 1, 2, 3)),
                 ([0, .001, .002, .1], (0, 1, 2)),
                 ([0, .01, .1, .101], (2, 3)),
                 ([0, .1, .2, .3], (0,)),
                 # Adjacent agreement does not imply all three agree.
                 ([0, .015, .03], (0, 1))]
        for offsets, expected in cases:
            with self.subTest(offsets=offsets):
                selected, _ = self.detector._select_consensus(self.poses(offsets))
                self.assertEqual(selected, expected)

    def test_fallback_without_top_left(self):
        poses = self.poses([0, .1, .2, .3])
        del poses[0]
        self.assertEqual(self.detector._select_consensus(poses), ((1,), "fallback"))

    def test_orientation_disagreement(self):
        poses = self.poses([0, 0])
        poses[1] = (np.array([0., 0., 0.8]), self.tvec)
        self.assertEqual(self.detector._select_consensus(poses)[1], "fallback")

    def test_rigid_recovery_and_outlier_rejection(self):
        result = self.detector._from_markers(*self.markers([0, 0, 0, .1]))
        self.assertTrue(result.success)
        self.assertEqual(result.num_tags_visible, 4)
        self.assertEqual(result.used_tag_ids, (0, 1, 2))
        self.assertEqual(result.rejected_tag_ids, (3,))
        self.assertEqual(len(result.locations), 87)
        self.assertFalse(result.ready_for_control)
        expected = (cv2.Rodrigues(self.rvec)[0] @ np.array(KEY_LOCATIONS_M["A"]) + self.tvec)
        np.testing.assert_allclose(result.locations["A"]["camera_xyz_m"], expected, atol=1e-7)
        self.assertLess(result.reprojection_error_px, 1e-6)

    def test_single_key_single_tag(self):
        result = self.detector._from_markers(*self.markers([0]), key="A")
        self.assertTrue(result.success)
        self.assertEqual(list(result.locations), ["A"])
        self.assertEqual(result.selection, "single_tag")
        self.assertFalse(result.ready_for_control)

    def test_missing_and_duplicate_tags(self):
        self.assertFalse(self.detector.detect(np.full((600, 800, 3), 255, np.uint8)).success)
        corners, ids = self.markers([0])
        result = self.detector._from_markers(corners*2, np.array([[0], [0]]))
        self.assertFalse(result.success)
        self.assertEqual(result.locations, {})
        with self.assertRaises(KeyError):
            self.detector.detect(np.zeros((10, 10), np.uint8), "unknown")

    def test_detect_rendered_tags(self):
        # Exercise actual image -> ArUco -> pose -> key projections.
        image = np.full((600, 800), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        tvec = np.array([-0.177, -0.0615, 0.7])
        for tag in self.detector.config.tags:
            target = cv2.projectPoints(tag.corners(), np.zeros(3), tvec,
                                       self.K, np.zeros(5))[0].reshape(4, 2).astype(np.float32)
            marker = cv2.aruco.generateImageMarker(dictionary, tag.tag_id, 160)
            source = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], np.float32)
            warped = cv2.warpPerspective(marker, cv2.getPerspectiveTransform(source, target),
                                         (800, 600), flags=cv2.INTER_NEAREST, borderValue=255)
            image = np.minimum(image, warped)
        result = self.detector.detect(image, "a")
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.num_tags_visible, 4)
        self.assertEqual(len(result.used_tag_ids), 4)
        expected = np.array(KEY_LOCATIONS_M["A"]) + tvec
        np.testing.assert_allclose(result.locations["A"]["camera_xyz_m"], expected, atol=.015)


if __name__ == "__main__":
    unittest.main()

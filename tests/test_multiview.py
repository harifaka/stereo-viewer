import unittest

import cv2
import numpy as np

import app as stereo_app
from multiview import (
    DEFAULT_SETTINGS,
    FrameHub,
    MultiViewRuntime,
    SlamState,
    match_features,
    relative_pose,
    render_mvs,
    render_slam,
    render_thermal,
    render_thermal_stereo,
)


def textured_image(shift=0):
    generator = np.random.default_rng(3)
    image = np.zeros((240, 320, 3), np.uint8)
    for index in range(140):
        center = (int(generator.integers(16, 304)), int(generator.integers(16, 224)))
        color = tuple(int(value) for value in generator.integers(30, 255, size=3))
        cv2.circle(image, center, int(generator.integers(4, 11)), color, -1)
        cv2.putText(
            image,
            str(index % 10),
            (center[0] - 4, center[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if shift:
        transform = np.array([[1, 0, shift], [0, 1, 0]], np.float32)
        image = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))
    return image


class MultiViewAlgorithmTests(unittest.TestCase):
    def test_feature_matches_and_fundamental_matrix_for_a_shifted_pair(self):
        reference = textured_image()
        other = textured_image(shift=10)
        matched = match_features(reference, other, 'ORB')

        self.assertIsNotNone(matched)
        self.assertGreaterEqual(int(np.count_nonzero(matched['inliers'])), 8)
        self.assertEqual(matched['fundamental'].shape, (3, 3))

    def test_mvs_reports_sparse_points_and_rejects_a_single_camera(self):
        reference = textured_image()
        other = textured_image(shift=8)
        settings = dict(DEFAULT_SETTINGS, referenceSlot=1, detector='ORB')
        image, status = render_mvs({1: reference, 2: other}, settings, None)

        self.assertTrue(status['ok'])
        self.assertGreaterEqual(status['inliers'], 8)
        self.assertGreater(status['points'], 0)
        self.assertEqual(image.shape[0], reference.shape[0])

        _, single = render_mvs({1: reference}, settings, None)
        self.assertFalse(single['ok'])

    def test_thermal_overlay_matches_the_rgb_frame_size(self):
        rgb = textured_image()
        thermal = textured_image(shift=6)
        settings = dict(DEFAULT_SETTINGS, rgbSlot=1, thermalSlots=[2], overlayOpacity=0.5, pictureInPicture=True)
        image, status = render_thermal({1: rgb, 2: thermal}, settings)

        self.assertTrue(status['ok'])
        self.assertTrue(status['aligned'][0]['ok'])
        self.assertEqual(image.shape[:2], rgb.shape[:2])

    def test_clahe_thermal_stereo_returns_a_disparity_image(self):
        left = textured_image()
        right = textured_image(shift=4)
        settings = dict(DEFAULT_SETTINGS, thermalLeftSlot=1, thermalRightSlot=2, claheClip=2.0)
        image, status = render_thermal_stereo(
            {1: left, 2: right},
            settings,
            stereo_app.stereo_params,
            stereo_app.COLOR_MAPS['TURBO'],
            stereo_app.STEREO_MODES['SGBM'],
        )

        self.assertTrue(status['ok'])
        self.assertEqual(image.shape[:2], left.shape[:2])

    def test_optical_flow_tracks_a_shifted_frame(self):
        state = SlamState()
        settings = dict(DEFAULT_SETTINGS, slamReferenceSlot=1)
        render_slam({1: textured_image()}, settings, state)
        _, status = render_slam({1: textured_image(shift=6)}, settings, state)

        self.assertTrue(status['ok'])
        self.assertGreater(status['tracked'], 0)
        self.assertGreater(status['pathLength'], 0)

    def test_identical_camera_poses_have_no_relative_motion(self):
        rotation_vector = np.array([[0.2], [-0.1], [0.05]], np.float64)
        translation = np.array([[10.0], [4.0], [30.0]], np.float64)
        rotation, relative_translation = relative_pose(rotation_vector, translation, rotation_vector, translation)

        self.assertTrue(np.allclose(rotation, np.eye(3), atol=1e-6))
        self.assertTrue(np.allclose(relative_translation, 0, atol=1e-6))

    def test_frame_hub_returns_an_independent_copy(self):
        hub = FrameHub()
        frame = np.zeros((4, 4, 3), np.uint8)
        hub.update('client-1234', {1: frame})
        frame[0, 0] = (255, 0, 0)
        snapshot = hub.snapshot('client-1234')

        self.assertEqual(int(snapshot[1][0, 0, 0]), 0)


class CharucoAndSettingsTests(unittest.TestCase):
    def setUp(self):
        self.runtime = stereo_app.multiview_runtime
        self.original_settings = self.runtime.current_settings()
        self.original_board = dict(self.runtime.board)
        self.client = stereo_app.app.test_client()

    def tearDown(self):
        self.runtime.update_settings(self.original_settings)
        self.runtime.update_board(self.original_board)
        self.runtime.reset_charuco()

    def test_detects_a_generated_board_and_rejects_too_few_views(self):
        board_image = self.runtime.board_image()
        bordered = cv2.copyMakeBorder(board_image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        frame = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        status = self.runtime.capture({1: frame, 2: frame})

        self.assertEqual(status['views'], 1)
        self.assertGreaterEqual(status['detections'][1], 1)
        with self.assertRaises(ValueError):
            self.runtime.calibrate()

    def test_board_endpoint_returns_a_png(self):
        response = self.client.get('/charuco-board.png')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/png')
        self.assertTrue(response.data.startswith(b'\x89PNG'))

    def test_rejects_an_unknown_processing_mode(self):
        response = self.client.post('/api/multiview/settings', json={'mode': 'mesh'})

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_multi_camera_page_exposes_the_mode_switcher(self):
        response = self.client.get('/multi-camera')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="processingMode"', response.data)
        self.assertIn(b'value="mvs"', response.data)
        self.assertIn(b'value="thermal-stereo"', response.data)
        self.assertIn(b'id="addCameraSlot"', response.data)
        self.assertIn(b'ChArUco', response.data)


if __name__ == '__main__':
    unittest.main()

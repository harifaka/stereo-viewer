import unittest

import cv2
import numpy as np

import app as stereo_app
import stereo_extras
import temporal
from test_multiview import textured_image


class OutputViewTests(unittest.TestCase):
    def test_anaglyph_takes_red_from_left_and_green_blue_from_right(self):
        left = np.full((4, 4, 3), (10, 20, 200), np.uint8)
        right = np.full((4, 4, 3), (90, 80, 5), np.uint8)
        image = stereo_extras.anaglyph(left, right)

        self.assertEqual(tuple(int(value) for value in image[0, 0]), (90, 80, 200))

    def test_bokeh_keeps_the_foreground_sharp(self):
        frame = textured_image()
        disparity = np.zeros(frame.shape[:2], np.float32)
        disparity[:, :160] = 40
        image = stereo_extras.bokeh(frame, disparity, 20, 31)

        foreground_error = np.abs(image[40:200, 20:120].astype(int) - frame[40:200, 20:120].astype(int)).mean()
        background_error = np.abs(image[40:200, 220:300].astype(int) - frame[40:200, 220:300].astype(int)).mean()
        self.assertLess(foreground_error, background_error)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([
            [1, 0, 0, -160],
            [0, 1, 0, -120],
            [0, 0, 0, 300],
            [0, 0, 1 / 60, 0],
        ], np.float64)
        self.disparity = np.full((240, 320), 30, np.float32)
        self.frame = textured_image()

    def test_point_cloud_ply_has_a_vertex_per_valid_pixel(self):
        points, colors, mask = stereo_extras.reproject(self.disparity, self.q, self.frame, 0)
        payload = stereo_extras.ply_point_cloud(points[mask], colors[mask])

        self.assertTrue(payload.startswith(b'ply\nformat binary_little_endian 1.0\n'))
        self.assertIn(f'element vertex {int(mask.sum())}'.encode(), payload)

    def test_grid_mesh_triangulates_a_flat_surface(self):
        points, colors, mask = stereo_extras.reproject(self.disparity, self.q, self.frame, 0)
        vertices, vertex_colors, faces = stereo_extras.grid_mesh(points, colors, mask, step=4)

        self.assertEqual(len(vertices), len(vertex_colors))
        self.assertGreater(len(faces), 0)
        self.assertLess(int(faces.max()), len(vertices))
        self.assertIn(b'element face', stereo_extras.ply_mesh(vertices, vertex_colors, faces))

    def test_export_requires_metric_calibration(self):
        _, encoded = cv2.imencode('.jpg', self.frame)
        client = stereo_app.app.test_client()
        response = client.post('/api/pointcloud', data={
            'left': (__import__('io').BytesIO(encoded.tobytes()), 'left.jpg'),
            'right': (__import__('io').BytesIO(encoded.tobytes()), 'right.jpg'),
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 409)


class TemporalTests(unittest.TestCase):
    def test_lucas_kanade_track_set_follows_a_shift(self):
        track_set = temporal.PointTrackSet(temporal.LucasKanadeBackend(), max_points=40, hold_frames=5)
        track_set.step(textured_image())
        before = track_set.positions.copy()
        ids = track_set.ids.copy()
        track_set.step(textured_image(shift=5))

        self.assertGreater(track_set.summary()['visible'], 10)
        common = np.isin(track_set.ids, ids)
        moved = track_set.positions[common][:, 0] - before[np.isin(ids, track_set.ids)][:, 0]
        self.assertAlmostEqual(float(np.median(moved)), 5.0, delta=1.0)

    def test_disparity_filter_fills_short_holes(self):
        gray = cv2.cvtColor(textured_image(), cv2.COLOR_BGR2GRAY)
        temporal_filter = temporal.DisparityTemporalFilter()
        first = np.full((240, 320), 20, np.float32)
        temporal_filter.apply(first, gray, 0, 0.5, 5)
        holes = first.copy()
        holes[100:120, 100:120] = -1
        output, stats = temporal_filter.apply(holes, gray, 0, 0.5, 5)

        self.assertGreater(stats['filledPixels'], 0)
        self.assertTrue(np.allclose(output[100:120, 100:120], 20))

    def test_drift_corrector_reduces_a_vertical_offset(self):
        left = textured_image()
        transform = np.array([[1, 0, -6], [0, 1, 4]], np.float32)
        right = cv2.warpAffine(left, transform, (left.shape[1], left.shape[0]))
        corrector = temporal.StereoDriftCorrector(interval=1, gain=1.0)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        for _ in range(4):
            corrected, _ = corrector.correct(gray_left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), right)

        self.assertAlmostEqual(corrector.status()['offsetPx'], -4.0, delta=1.0)
        self.assertEqual(corrected.shape, right.shape)

    def test_tap_runtime_renders_selected_targets(self):
        runtime = temporal.TapRuntime()
        settings = {
            'enabled': True, 'cameraSlots': [1], 'composite': True,
            'backend': 'lucas-kanade', 'maxPoints': 30, 'holdFrames': 5,
        }
        runtime.render({1: textured_image(), 2: textured_image(shift=3)}, settings)
        _, status = runtime.render({1: textured_image(shift=2), 2: textured_image(shift=5)}, settings)

        self.assertTrue(status['ok'])
        self.assertEqual([target['target'] for target in status['targets']], ['camera-1', 'composite'])


class SettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = stereo_app.app.test_client()

    def tearDown(self):
        self.client.post('/api/output-settings', json={'view': 'disparity', 'bokehThreshold': 0.45, 'bokehBlur': 31})
        self.client.post('/api/detection-settings', json={'enabled': False})
        self.client.post('/api/temporal-settings', json={'enabled': False, 'autoCorrect': False, 'backend': 'auto'})

    def test_output_view_validation(self):
        self.assertEqual(self.client.post('/api/output-settings', json={'view': 'anaglyph'}).get_json()['view'], 'anaglyph')
        self.assertEqual(self.client.post('/api/output-settings', json={'view': 'xray'}).status_code, 400)
        self.assertEqual(self.client.post('/api/output-settings', json={'bokehBlur': 30}).status_code, 400)

    def test_temporal_settings_report_drift_and_backend(self):
        response = self.client.post('/api/temporal-settings', json={'enabled': True, 'backend': 'lucas-kanade'})

        self.assertEqual(response.status_code, 200)
        self.assertIn('drift', response.get_json())
        self.assertIn('tapirAvailable', response.get_json())

    def test_live_disparity_publishes_the_selected_view(self):
        self.client.post('/api/output-settings', json={'view': 'anaglyph'})
        left = textured_image()
        right = textured_image(shift=4)
        response = self.client.post('/api/disparity', data={
            'left': (__import__('io').BytesIO(cv2.imencode('.jpg', left)[1].tobytes()), 'left.jpg'),
            'right': (__import__('io').BytesIO(cv2.imencode('.jpg', right)[1].tobytes()), 'right.jpg'),
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Output-View'], 'anaglyph')
        self.assertIn(response.headers['X-Stereo-Backend'], ('cpu', 'cuda'))

    def test_acceleration_endpoint(self):
        result = self.client.get('/api/acceleration').get_json()

        self.assertIn('cudaStereoAvailable', result)
        self.assertIn('inferenceDevice', result)


if __name__ == '__main__':
    unittest.main()

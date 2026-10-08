import unittest

import app as stereo_app


class TapSettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.original_settings = stereo_app.tap_settings.copy()
        self.client = stereo_app.app.test_client()

    def tearDown(self):
        with stereo_app.tap_settings_lock:
            stereo_app.tap_settings = self.original_settings

    def test_saves_camera_and_composite_targets_as_configuration_only(self):
        response = self.client.post(
            '/api/tap-settings',
            json={'enabled': True, 'cameraSlots': [1, 3], 'composite': True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                'enabled': True,
                'cameraSlots': [1, 3],
                'composite': True,
                'available': False,
                'status': 'configuration_only',
            },
        )

    def test_get_reports_tracking_as_not_installed(self):
        response = self.client.get('/api/tap-settings')

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()['available'])
        self.assertEqual(response.get_json()['status'], 'configuration_only')

    def test_rejects_duplicate_or_out_of_range_camera_slots(self):
        for camera_slots in ([1, 1], [0], [5], [True]):
            with self.subTest(camera_slots=camera_slots):
                response = self.client.post(
                    '/api/tap-settings',
                    json={
                        'enabled': True,
                        'cameraSlots': camera_slots,
                        'composite': False,
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.get_json())

    def test_rejects_enabled_tracking_without_targets(self):
        response = self.client.post(
            '/api/tap-settings',
            json={'enabled': True, 'cameraSlots': [], 'composite': False},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()['error'],
            'Select at least one camera or the composite view.',
        )

    def test_renders_the_multi_camera_page(self):
        response = self.client.get('/multi-camera')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Multi-Camera Monitor', response.data)
        self.assertIn(b'TAP-Net temporal tracking', response.data)
        self.assertIn(b'href="/" data-copy="stereoPage"', response.data)
        self.assertIn(b'href="/multi-camera" aria-current="page"', response.data)
        self.assertIn(b'data-camera-settings-open', response.data)
        self.assertIn(b'cameraSettingsDialog', response.data)
        self.assertIn(b'/api/cameras/start', response.data)

    def test_renders_page_navigation_on_the_stereo_homepage(self):
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'href="/" aria-current="page"', response.data)
        self.assertIn(b'href="/multi-camera"', response.data)
        self.assertIn(b'data-camera-settings-open', response.data)
        self.assertIn(b'cameraCaptureSource', response.data)

    def test_redirects_the_misspelled_multi_camera_url(self):
        response = self.client.get('/multy-camera')

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, '/multi-camera')


if __name__ == '__main__':
    unittest.main()

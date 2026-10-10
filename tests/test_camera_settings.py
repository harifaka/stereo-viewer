import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as stereo_app


def camera_configuration():
    return {
        'cameras': [
            {'slot': slot, 'label': f'Camera {slot}', 'deviceIndex': slot - 1, 'active': slot <= 2}
            for slot in range(1, 5)
        ]
    }


class CameraSettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_path = stereo_app.CAMERA_SETTINGS_PATH
        self.original_settings = stereo_app.camera_settings
        self.original_manager = stereo_app.camera_stream_manager
        stereo_app.CAMERA_SETTINGS_PATH = Path(self.temporary_directory.name) / 'cameras.json'
        stereo_app.camera_settings = camera_configuration()
        stereo_app.camera_stream_manager = stereo_app.CameraStreamManager()
        self.client = stereo_app.app.test_client()

    def tearDown(self):
        stereo_app.camera_stream_manager.stop()
        stereo_app.CAMERA_SETTINGS_PATH = self.original_path
        stereo_app.camera_settings = self.original_settings
        stereo_app.camera_stream_manager = self.original_manager
        self.temporary_directory.cleanup()

    def test_get_returns_four_default_camera_slots(self):
        response = self.client.get('/api/camera-settings')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), camera_configuration())

    def test_post_persists_labels_indexes_and_active_flags(self):
        settings = camera_configuration()
        settings['cameras'][0].update(label='Front left', deviceIndex=3, active=False)
        settings['cameras'][3]['deviceIndex'] = 0

        response = self.client.post('/api/camera-settings', json=settings)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), settings)
        with stereo_app.CAMERA_SETTINGS_PATH.open(encoding='utf-8') as config_file:
            self.assertEqual(json.load(config_file), settings)
        self.assertEqual(self.client.get('/api/camera-settings').get_json(), settings)

    def test_saving_default_settings_creates_the_configuration_file(self):
        response = self.client.post('/api/camera-settings', json=camera_configuration())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(stereo_app.CAMERA_SETTINGS_PATH.is_file())

    def test_rejects_duplicate_or_invalid_indexes_and_slots(self):
        invalid_cases = []
        duplicated = camera_configuration()
        duplicated['cameras'][1]['deviceIndex'] = 0
        invalid_cases.append(duplicated)
        boolean_index = camera_configuration()
        boolean_index['cameras'][0]['deviceIndex'] = True
        invalid_cases.append(boolean_index)
        missing_slot = camera_configuration()
        missing_slot['cameras'][0]['slot'] = 2
        invalid_cases.append(missing_slot)

        for settings in invalid_cases:
            with self.subTest(settings=settings):
                response = self.client.post('/api/camera-settings', json=settings)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.get_json())

    def test_persists_one_camera_and_more_than_four(self):
        cases = [
            [{
                'slot': 1,
                'label': 'Only',
                'deviceIndex': 31,
                'active': True,
            }],
            [
                {
                    'slot': slot,
                    'label': f'Camera {slot}',
                    'deviceIndex': slot - 1,
                    'active': slot == 1,
                }
                for slot in range(1, 7)
            ],
        ]
        for cameras in cases:
            settings = {'cameras': cameras}
            with self.subTest(cameras=len(cameras)):
                response = self.client.post('/api/camera-settings', json=settings)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), settings)
                self.assertEqual(self.client.get('/api/camera-settings').get_json(), settings)

    def test_rejects_empty_too_many_and_out_of_range_indexes(self):
        too_many = {
            'cameras': [
                {
                    'slot': slot,
                    'label': f'Camera {slot}',
                    'deviceIndex': slot - 1,
                    'active': False,
                }
                for slot in range(1, 18)
            ]
        }
        out_of_range = camera_configuration()
        out_of_range['cameras'][0]['deviceIndex'] = 32
        negative = camera_configuration()
        negative['cameras'][0]['deviceIndex'] = -1
        for settings in ({'cameras': []}, too_many, out_of_range, negative):
            with self.subTest(settings=settings):
                response = self.client.post('/api/camera-settings', json=settings)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.get_json())

    def test_requires_a_camera_client_id_before_starting_server_cameras(self):
        response = self.client.post('/api/cameras/start', json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'A valid camera client ID is required.')

    def test_available_camera_endpoint_lists_detected_video_nodes(self):
        original_devices = stereo_app.AVAILABLE_CAMERA_INDICES
        stereo_app.AVAILABLE_CAMERA_INDICES = [0, 4, 31]
        try:
            response = self.client.get('/api/cameras/available')
        finally:
            stereo_app.AVAILABLE_CAMERA_INDICES = original_devices

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            'devices': [
                {'deviceIndex': 0, 'path': '/dev/video0'},
                {'deviceIndex': 4, 'path': '/dev/video4'},
                {'deviceIndex': 31, 'path': '/dev/video31'},
            ]
        })


class CameraDiscoveryTests(unittest.TestCase):
    def test_detects_only_existing_devices_that_return_a_frame(self):
        class FakeCapture:
            def __init__(self, device_index):
                self.device_index = device_index

            def isOpened(self):
                return self.device_index in (1, 2, 3)

            def read(self):
                return self.device_index == 1, object() if self.device_index == 1 else None

            def release(self):
                pass

        def exists(path):
            return path in ('/dev/video1', '/dev/video2', '/dev/video3')

        with patch.object(stereo_app.os.path, 'exists', side_effect=exists), patch.object(
            stereo_app.cv2, 'VideoCapture', side_effect=FakeCapture
        ):
            self.assertEqual(stereo_app.detect_available_cameras(), [1])

    def test_automatic_settings_assign_the_first_three_detected_devices(self):
        settings = stereo_app.automatic_camera_settings([2, 5, 9, 12])

        self.assertEqual(
            [(camera['slot'], camera['deviceIndex'], camera['active']) for camera in settings['cameras']],
            [(1, 2, True), (2, 5, True), (3, 9, True)],
        )

    def test_outdated_settings_fall_back_to_detected_devices(self):
        original_path = stereo_app.CAMERA_SETTINGS_PATH
        with tempfile.TemporaryDirectory() as directory:
            stereo_app.CAMERA_SETTINGS_PATH = Path(directory) / 'cameras.json'
            with stereo_app.CAMERA_SETTINGS_PATH.open('w', encoding='utf-8') as config_file:
                json.dump(camera_configuration(), config_file)
            try:
                settings = stereo_app.initial_camera_settings([6, 8, 12])
            finally:
                stereo_app.CAMERA_SETTINGS_PATH = original_path

        self.assertEqual(
            [camera['deviceIndex'] for camera in settings['cameras']],
            [6, 8, 12],
        )


class CameraStreamManagerTests(unittest.TestCase):
    def test_shares_capture_workers_until_the_last_client_disconnects(self):
        created_streams = []

        class FakeCameraStream:
            def __init__(self, slot, label, device_index):
                self.slot = slot
                self.stopped = False
                created_streams.append(self)

            def start(self):
                pass

            def request_stop(self):
                pass

            def join(self):
                self.stopped = True

        manager = stereo_app.CameraStreamManager()
        with patch.object(stereo_app, 'CameraStream', FakeCameraStream):
            manager.start(camera_configuration(), 'client-one')
            manager.start(camera_configuration(), 'client-two')

            self.assertEqual(len(created_streams), 2)
            manager.stop('client-one')
            self.assertFalse(any(stream.stopped for stream in created_streams))
            manager.stop('client-two')

        self.assertTrue(all(stream.stopped for stream in created_streams))
        self.assertEqual(manager.status(), [])


if __name__ == '__main__':
    unittest.main()

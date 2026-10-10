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

    def test_multi_camera_settings_panel_exposes_manual_rescan(self):
        response = self.client.get('/multi-camera')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="rescanCameraDevices"', response.data)

    def test_stereo_monitor_offers_browser_and_docker_capture_sources(self):
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="stereoCaptureSource"', response.data)
        self.assertIn(b'value="docker"', response.data)
        self.assertIn(b'id="leftDockerImage"', response.data)

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
        with patch.object(stereo_app, 'detect_available_cameras', return_value=[0, 4, 31]):
            response = self.client.get('/api/cameras/available')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            'devices': [
                {'deviceIndex': 0, 'path': '/dev/video0'},
                {'deviceIndex': 4, 'path': '/dev/video4'},
                {'deviceIndex': 31, 'path': '/dev/video31'},
            ]
        })

    def test_rescan_reassigns_detected_cameras_and_persists_the_layout(self):
        with patch.object(stereo_app, 'detect_available_cameras', return_value=[4, 6, 9, 12]):
            response = self.client.post('/api/cameras/rescan')

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result['assigned'])
        self.assertEqual(
            [camera['deviceIndex'] for camera in result['settings']['cameras']],
            [4, 6, 9],
        )
        self.assertEqual(
            [device['path'] for device in result['devices']],
            ['/dev/video4', '/dev/video6', '/dev/video9', '/dev/video12'],
        )
        with stereo_app.CAMERA_SETTINGS_PATH.open(encoding='utf-8') as config_file:
            self.assertEqual(json.load(config_file), result['settings'])

    def test_rescan_with_no_working_devices_keeps_current_assignments(self):
        with patch.object(stereo_app, 'detect_available_cameras', return_value=[]):
            response = self.client.post('/api/cameras/rescan')

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()['assigned'])
        self.assertEqual(response.get_json()['settings'], camera_configuration())


class CameraDiscoveryTests(unittest.TestCase):
    def test_detects_only_existing_devices_that_return_a_frame(self):
        released_devices = []
        capture_backends = []
        read_attempts = {}

        class FakeCapture:
            def __init__(self, device_index, backend):
                self.device_index = device_index
                capture_backends.append(backend)

            def isOpened(self):
                return self.device_index in (1, 2, 3)

            def read(self):
                read_attempts[self.device_index] = read_attempts.get(self.device_index, 0) + 1
                return self.device_index == 1, object() if self.device_index == 1 else None

            def release(self):
                released_devices.append(self.device_index)

        def exists(path):
            return path in ('/dev/video1', '/dev/video2', '/dev/video3')

        with patch.object(stereo_app.os.path, 'exists', side_effect=exists), patch.object(
            stereo_app.cv2, 'VideoCapture', side_effect=FakeCapture
        ):
            self.assertEqual(stereo_app.detect_available_cameras(), [1])

        self.assertEqual(released_devices, [1, 2, 3])
        self.assertEqual(capture_backends, [stereo_app.cv2.CAP_V4L2] * 3)
        self.assertEqual(read_attempts, {1: 1, 2: 5, 3: 5})

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
    def test_keeps_retrying_a_camera_that_is_not_openable(self):
        stream = stereo_app.CameraStream(1, 'Camera 1', 0)

        class FakeCapture:
            def __init__(self):
                self.released = False

            def isOpened(self):
                stream._stop_event.set()
                return False

            def release(self):
                self.released = True

        capture = FakeCapture()
        with patch.object(stereo_app.cv2, 'VideoCapture', return_value=capture), patch.object(
            stereo_app.app.logger, 'warning'
        ) as warning:
            stream._capture()

        self.assertTrue(capture.released)
        self.assertIn('retry', warning.call_args.args[0])

    def test_reopens_after_a_transient_camera_open_failure(self):
        stream = stereo_app.CameraStream(1, 'Camera 1', 0)
        captures = []

        class FakeFrame:
            def copy(self):
                return self

        class FakeBuffer:
            def tobytes(self):
                return b'jpeg'

        class FakeCapture:
            def __init__(self, device_index, backend):
                self.attempt = len(captures) + 1
                self.released = False
                captures.append(self)

            def isOpened(self):
                return self.attempt > 1

            def read(self):
                stream._stop_event.set()
                return True, FakeFrame()

            def release(self):
                self.released = True

        with patch.object(stereo_app.cv2, 'VideoCapture', side_effect=FakeCapture), patch.object(
            stereo_app.cv2, 'imencode', return_value=(True, FakeBuffer())
        ), patch.object(stream._stop_event, 'wait', return_value=False):
            stream._capture()

        self.assertEqual(len(captures), 2)
        self.assertTrue(all(capture.released for capture in captures))
        self.assertEqual(stream.status()['state'], 'disconnected')
        self.assertIsNone(stream.status()['error'])
        self.assertEqual(stream._frame, b'jpeg')

    def test_rescan_releases_running_captures_before_discovery_and_restarts_them(self):
        created_streams = []

        class FakeCameraStream:
            def __init__(self, slot, label, device_index):
                self.slot = slot
                self.label = label
                self.device_index = device_index
                self.stopped = False
                created_streams.append(self)

            def start(self):
                pass

            def request_stop(self):
                self.stopped = True

            def join(self):
                pass

        manager = stereo_app.CameraStreamManager()
        with patch.object(stereo_app, 'CameraStream', FakeCameraStream):
            manager.start(camera_configuration(), 'client-one')
            old_streams = list(created_streams)
            updated_settings = stereo_app.automatic_camera_settings([4, 6, 9])

            def discover():
                self.assertTrue(all(stream.stopped for stream in old_streams))
                self.assertEqual(manager._streams, {})
                return updated_settings

            self.assertEqual(manager.rescan(discover), updated_settings)

        self.assertEqual(
            [stream.device_index for stream in created_streams[len(old_streams):]],
            [4, 6, 9],
        )
        self.assertEqual(manager._clients, {'client-one'})
        manager.stop()

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

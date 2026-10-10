import json
import logging
import math
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread

from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for
import cv2
import numpy as np
import multiview
import stereo_extras
import temporal
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
stereo_params = {
    'minDisparity': 0,
    'numDisparities': 64,
    'blockSize': 5,
    'uniquenessRatio': 10,
    'speckleWindowSize': 50,
    'speckleRange': 2,
    'disp12MaxDiff': 1,
    'preFilterCap': 31,
    'mode': '3WAY',
    'colorMap': 'TURBO',
}
params_lock = Lock()
calibration_lock = Lock()
calibration_state = {'captures': [], 'board': None, 'active': None, 'method': None}
pose_settings_lock = Lock()
pose_model_lock = Lock()
pose_model = None
pose_settings = {'enabled': False, 'view': 'composite', 'confidence': 0.35}
tap_settings_lock = Lock()
tap_settings = {
    'enabled': False,
    'cameraSlots': [1, 2],
    'composite': True,
    'backend': 'auto',
    'maxPoints': 80,
    'holdFrames': 12,
}
output_settings_lock = Lock()
output_settings = {'view': 'disparity', 'bokehThreshold': 0.45, 'bokehBlur': 31}
detection_settings_lock = Lock()
detection_model_lock = Lock()
detection_model = None
detection_settings = {'enabled': False, 'view': 'both', 'confidence': 0.4}
temporal_settings_lock = Lock()
temporal_settings = {
    'enabled': False,
    'smoothing': 0.6,
    'holdFrames': 10,
    'trackMeasurement': True,
    'autoCorrect': False,
    'backend': 'auto',
}
temporal_lock = Lock()
disparity_filter = temporal.DisparityTemporalFilter()
measurement_tracker = temporal.MeasurementTracker()
drift_corrector = temporal.StereoDriftCorrector()
temporal_backend_message = ''
temporal_backend_preference = None
stereo_output_hub = stereo_extras.OutputHub()
tap_runtime = temporal.TapRuntime()
acceleration_lock = Lock()
acceleration_state = {'stereoBackend': 'cpu', 'cudaError': None}
CAMERA_SETTINGS_PATH = Path(os.environ.get('CAMERA_SETTINGS_PATH', 'config/cameras.json'))
DEFAULT_CAMERA_SETTINGS = {
    'cameras': [
        {
            'slot': slot,
            'label': f'Camera {slot}',
            'deviceIndex': slot - 1,
            'active': slot <= 2,
        }
        for slot in range(1, 5)
    ]
}


def detect_available_cameras():
    available = []
    for device_index in range(32):
        device_path = f'/dev/video{device_index}'
        if not os.path.exists(device_path):
            continue

        capture = None
        try:
            capture = cv2.VideoCapture(device_index)
            if not capture.isOpened():
                continue
            success, frame = capture.read()
            if success and frame is not None:
                available.append(device_index)
        except cv2.error:
            app.logger.exception('Could not probe camera device %s.', device_path)
        finally:
            if capture is not None:
                capture.release()
    app.logger.info('Detected functional video devices: %s', available)
    return available


def automatic_camera_settings(device_indices):
    if not device_indices:
        return DEFAULT_CAMERA_SETTINGS
    return {
        'cameras': [
            {
                'slot': slot,
                'label': f'Camera {slot}',
                'deviceIndex': device_index,
                'active': True,
            }
            for slot, device_index in enumerate(device_indices[:3], start=1)
        ]
    }


def validate_camera_settings(data):
    if not isinstance(data, dict) or not isinstance(data.get('cameras'), list):
        raise ValueError('Camera settings must contain a cameras list.')
    camera_count = len(data['cameras'])
    if camera_count < 1 or camera_count > 16:
        raise ValueError('Configure between 1 and 16 camera slots.')

    cameras = []
    device_indices = set()
    for expected_slot, camera in enumerate(data['cameras'], start=1):
        if (
            not isinstance(camera, dict)
            or type(camera.get('slot')) is not int
            or camera.get('slot') != expected_slot
        ):
            raise ValueError('Camera slots must be numbered 1 through N in order.')
        label = camera.get('label')
        if not isinstance(label, str) or not label.strip() or len(label.strip()) > 40:
            raise ValueError('Each camera label must contain 1 to 40 characters.')
        device_index = camera.get('deviceIndex')
        if type(device_index) is not int or device_index not in range(32):
            raise ValueError('Camera device indexes must be integers from 0 to 31.')
        if device_index in device_indices:
            raise ValueError('Each camera must use a different device index.')
        device_indices.add(device_index)
        active = camera.get('active')
        if not isinstance(active, bool):
            raise ValueError('Each camera active value must be a boolean.')
        cameras.append({
            'slot': expected_slot,
            'label': label.strip(),
            'deviceIndex': device_index,
            'active': active,
        })
    return {'cameras': cameras}


def load_camera_settings():
    with CAMERA_SETTINGS_PATH.open(encoding='utf-8') as config_file:
        return validate_camera_settings(json.load(config_file))


def save_camera_settings(settings):
    CAMERA_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=CAMERA_SETTINGS_PATH.parent,
            delete=False,
        ) as config_file:
            temporary_path = Path(config_file.name)
            json.dump(settings, config_file, indent=2)
            config_file.write('\n')
        os.replace(temporary_path, CAMERA_SETTINGS_PATH)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        app.logger.exception('Could not save camera settings to %s', CAMERA_SETTINGS_PATH)
        raise


class CameraStream:
    def __init__(self, slot, label, device_index):
        self.slot = slot
        self.label = label
        self.device_index = device_index
        self._condition = Condition()
        self._stop_event = Event()
        self._frame = None
        self._raw_frame = None
        self._state = 'starting'
        self._error = None
        self._thread = Thread(target=self._capture, name=f'camera-{slot}', daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.request_stop()
        self.join()

    def request_stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

    def join(self):
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def status(self):
        with self._condition:
            return {
                'slot': self.slot,
                'label': self.label,
                'deviceIndex': self.device_index,
                'state': self._state,
                'error': self._error,
            }

    def frames(self):
        last_frame = None
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame is not None and self._frame is not last_frame
                    or self._stop_event.is_set(),
                    timeout=1,
                )
                frame = self._frame
                state = self._state
            if self._stop_event.is_set():
                break
            if frame is None or frame is last_frame:
                if state == 'error':
                    break
                continue
            last_frame = frame
            yield (
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + frame
                + b'\r\n'
            )

    def latest_frame(self):
        with self._condition:
            if self._raw_frame is None:
                return None
            return self._raw_frame.copy()

    def _capture(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.device_index)
            if not capture.isOpened():
                raise RuntimeError(f'Could not open /dev/video{self.device_index}.')
            with self._condition:
                self._state = 'connected'
                self._condition.notify_all()
            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f'Could not read from /dev/video{self.device_index}.')
                encoded, buffer = cv2.imencode('.jpg', frame)
                if not encoded:
                    raise RuntimeError(f'Could not encode a frame from /dev/video{self.device_index}.')
                with self._condition:
                    self._frame = buffer.tobytes()
                    self._raw_frame = frame.copy()
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._state = 'error'
                self._error = str(error)
                self._condition.notify_all()
            app.logger.exception(
                'Camera slot %s failed for /dev/video%s',
                self.slot,
                self.device_index,
            )
        finally:
            if capture is not None:
                capture.release()
            with self._condition:
                if self._state != 'error':
                    self._state = 'disconnected'
                self._condition.notify_all()


class CameraStreamManager:
    def __init__(self):
        self._lock = Lock()
        self._streams = {}
        self._running = False
        self._clients = set()

    def start(self, settings, client_id):
        with self._lock:
            if client_id not in self._clients:
                if not self._running:
                    self._running = True
                    self._replace_streams(settings)
                self._clients.add(client_id)

    def stop(self, client_id=None):
        with self._lock:
            if client_id is not None:
                self._clients.discard(client_id)
                if self._clients:
                    return
            else:
                self._clients.clear()
            streams = list(self._streams.values())
            self._streams = {}
            self._running = False
            for stream in streams:
                stream.request_stop()
            for stream in streams:
                stream.join()

    def configure(self, settings):
        with self._lock:
            was_running = self._running
            old_streams = list(self._streams.values()) if was_running else []
            if was_running:
                for stream in old_streams:
                    stream.request_stop()
                for stream in old_streams:
                    stream.join()
                self._streams = {}
                self._replace_streams(settings)

    def _replace_streams(self, settings):
        self._streams = {
            camera['slot']: CameraStream(
                camera['slot'],
                camera['label'],
                camera['deviceIndex'],
            )
            for camera in settings['cameras']
            if camera['active']
        }
        for stream in self._streams.values():
            stream.start()

    def stream(self, slot):
        with self._lock:
            return self._streams.get(slot)

    def status(self):
        with self._lock:
            return [stream.status() for stream in self._streams.values()]

    def latest_frames(self):
        with self._lock:
            streams = list(self._streams.values())
        frames = {}
        for stream in streams:
            frame = stream.latest_frame()
            if frame is not None:
                frames[stream.slot] = frame
        return frames


def initial_camera_settings(available_device_indices):
    if CAMERA_SETTINGS_PATH.is_file():
        try:
            settings = load_camera_settings()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            app.logger.warning(
                'Ignoring missing or outdated camera settings at %s: %s',
                CAMERA_SETTINGS_PATH,
                error,
            )
        else:
            if not available_device_indices or all(
                camera['deviceIndex'] in available_device_indices
                for camera in settings['cameras']
            ):
                return settings
            app.logger.info(
                'Camera settings reference devices that are not currently functional; '
                'using automatic camera assignments.'
            )
    return automatic_camera_settings(available_device_indices)


AVAILABLE_CAMERA_INDICES = detect_available_cameras()
camera_settings_lock = Lock()
camera_settings = initial_camera_settings(AVAILABLE_CAMERA_INDICES)
camera_stream_manager = CameraStreamManager()
multiview_runtime = multiview.MultiViewRuntime()
CALIBRATION_VIEWS_REQUIRED = 8
COLOR_MAPS = {
    'TURBO': cv2.COLORMAP_TURBO,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'INFERNO': cv2.COLORMAP_INFERNO,
}
STEREO_MODES = {
    'SGBM': cv2.STEREO_SGBM_MODE_SGBM,
    '3WAY': cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}
def decode_frame(upload):
    if upload is None:
        return None
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def validate_calibration_params(data, current_params):
    if not isinstance(data, dict):
        raise ValueError('Calibration settings must be a JSON object.')
    params = current_params.copy()
    integer_ranges = {
        'minDisparity': (-64, 64),
        'numDisparities': (16, 256),
        'blockSize': (3, 21),
        'uniquenessRatio': (0, 30),
        'speckleWindowSize': (0, 200),
        'speckleRange': (0, 32),
        'disp12MaxDiff': (-1, 64),
        'preFilterCap': (1, 63),
    }
    for name, (minimum, maximum) in integer_ranges.items():
        raw_value = data.get(name, params[name])
        if isinstance(raw_value, bool) or isinstance(raw_value, float):
            raise ValueError(f'{name} must be an integer.')
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError('Calibration values must be integers.') from None
        if value < minimum or value > maximum:
            raise ValueError(f'{name} must be between {minimum} and {maximum}.')
        params[name] = value
    if params['numDisparities'] % 16:
        raise ValueError('numDisparities must be a multiple of 16.')
    if params['blockSize'] % 2 == 0:
        raise ValueError('blockSize must be odd.')
    params['mode'] = data.get('mode', params['mode'])
    if not isinstance(params['mode'], str) or params['mode'] not in STEREO_MODES:
        raise ValueError('mode must be SGBM or 3WAY.')
    params['colorMap'] = data.get('colorMap', params['colorMap'])
    if not isinstance(params['colorMap'], str) or params['colorMap'] not in COLOR_MAPS:
        raise ValueError('colorMap must be TURBO, VIRIDIS, or INFERNO.')
    return params


def estimate_point(disparity_map, x, y, active, params):
    if not active or active['image_size'] != (disparity_map.shape[1], disparity_map.shape[0]):
        return None, 'Calibrate this camera resolution before measuring distance.'
    if not active.get('metric', True):
        return None, 'Feature alignment estimates relative disparity only; checkerboard calibration is required for metric distance.'
    if x < 0 or y < 0 or x >= disparity_map.shape[1] or y >= disparity_map.shape[0]:
        return None, 'Selected point is outside the disparity image.'
    region = disparity_map[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    valid = region[np.isfinite(region) & (region > params['minDisparity'])]
    if valid.size < 3:
        return None, 'No reliable stereo match at this point.'
    disparity = float(np.median(valid))
    homogeneous = active['q'] @ np.array([x, y, disparity, 1.0])
    if abs(homogeneous[3]) < 1e-9:
        return None, 'Distance is outside the calibrated range.'
    point = homogeneous[:3] / homogeneous[3]
    distance_m = float(np.linalg.norm(point) / 1000)
    if not math.isfinite(distance_m) or distance_m <= 0:
        return None, 'No reliable distance could be calculated at this point.'
    return {'x': x, 'y': y, 'disparity': round(disparity, 2), 'distanceM': round(distance_m, 3)}, None


def opencv_cuda_devices():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount()
    except (AttributeError, cv2.error):
        return 0


CUDA_STEREO_AVAILABLE = opencv_cuda_devices() > 0 and hasattr(cv2.cuda, 'createStereoSGM')
CUDA_SGM_DISPARITIES = (64, 128, 256)


def torch_cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def compute_disparity(gray_left, gray_right, params):
    """Runs StereoSGM on the GPU when OpenCV has CUDA support, otherwise StereoSGBM on the CPU."""
    block_size = params['blockSize']
    p1 = 8 * block_size * block_size
    p2 = 32 * block_size * block_size
    if CUDA_STEREO_AVAILABLE and params['numDisparities'] in CUDA_SGM_DISPARITIES:
        try:
            matcher = cv2.cuda.createStereoSGM(
                minDisparity=params['minDisparity'],
                numDisparities=params['numDisparities'],
                P1=p1,
                P2=p2,
                uniquenessRatio=params['uniquenessRatio'],
                mode=cv2.STEREO_SGBM_MODE_HH4,
            )
            gpu_left = cv2.cuda_GpuMat()
            gpu_right = cv2.cuda_GpuMat()
            gpu_left.upload(gray_left)
            gpu_right.upload(gray_right)
            disparity = matcher.compute(gpu_left, gpu_right).download().astype(np.float32) / 16
            with acceleration_lock:
                acceleration_state['stereoBackend'] = 'cuda'
            return disparity, 'cuda'
        except cv2.error as error:
            with acceleration_lock:
                if acceleration_state['cudaError'] is None:
                    app.logger.warning('CUDA StereoSGM failed; falling back to CPU: %s', error)
                acceleration_state['cudaError'] = str(error).splitlines()[0][:240]
    stereo = cv2.StereoSGBM_create(
        minDisparity=params['minDisparity'],
        numDisparities=params['numDisparities'],
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=params['disp12MaxDiff'],
        preFilterCap=params['preFilterCap'],
        uniquenessRatio=params['uniquenessRatio'],
        speckleWindowSize=params['speckleWindowSize'],
        speckleRange=params['speckleRange'],
        mode=STEREO_MODES[params['mode']],
    )
    with acceleration_lock:
        acceleration_state['stereoBackend'] = 'cpu'
    return stereo.compute(gray_left, gray_right).astype(np.float32) / 16, 'cpu'


def rectify_pair(frame_left, frame_right):
    """Returns rectified frames, the active calibration, and that calibration only if it fits this resolution."""
    with calibration_lock:
        active_calibration = calibration_state['active']
    if active_calibration and active_calibration['image_size'] == (frame_left.shape[1], frame_left.shape[0]):
        map_left = active_calibration['map_left']
        map_right = active_calibration['map_right']
        frame_left = cv2.remap(frame_left, map_left[0], map_left[1], cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right, map_right[0], map_right[1], cv2.INTER_LINEAR)
        return frame_left, frame_right, active_calibration, active_calibration
    return frame_left, frame_right, active_calibration, None


def colorize_disparity(disparity_map, params):
    valid = disparity_map > params['minDisparity']
    disparity_gray = np.zeros(disparity_map.shape, dtype=np.uint8)
    disparity_gray[valid] = np.clip(
        (disparity_map[valid] - params['minDisparity'])
        * (255 / params['numDisparities']),
        0,
        255,
    ).astype(np.uint8)
    color_disparity = cv2.applyColorMap(disparity_gray, COLOR_MAPS[params['colorMap']])
    color_disparity[~valid] = (0, 0, 0)
    return color_disparity


def render_output_view(rect_left, rect_right, disparity_map, params, options):
    if options['view'] == 'anaglyph':
        return stereo_extras.anaglyph(rect_left, rect_right)
    if options['view'] == 'bokeh':
        threshold = params['minDisparity'] + options['bokehThreshold'] * params['numDisparities']
        return stereo_extras.bokeh(rect_left, disparity_map, threshold, options['bokehBlur'])
    return colorize_disparity(disparity_map, params)


def run_temporal_stage(rect_left, gray_left, disparity_map, measurement_point, params, options):
    global temporal_backend_message, temporal_backend_preference
    with temporal_lock:
        if options['backend'] != temporal_backend_preference:
            backend, temporal_backend_message = temporal.create_backend(options['backend'])
            measurement_tracker.use_backend(backend)
            temporal_backend_preference = options['backend']
        disparity_map, filter_stats = disparity_filter.apply(
            disparity_map, gray_left, params['minDisparity'], options['smoothing'], options['holdFrames']
        )
        tracked_point = None
        if measurement_point is not None and options['trackMeasurement']:
            try:
                tracked_point = measurement_tracker.update(measurement_point, rect_left, options['holdFrames'])
            except Exception as error:
                app.logger.exception('measurement tracking failed')
                measurement_tracker.use_backend(temporal.LucasKanadeBackend())
                temporal_backend_message = f'Tracking failed ({error}); using Lucas-Kanade fallback.'[:240]
                tracked_point = {'x': measurement_point[0], 'y': measurement_point[1], 'tracked': False, 'occluded': False}
        data = {
            'backend': measurement_tracker.backend.name,
            'message': temporal_backend_message,
            **filter_stats,
        }
    return disparity_map, tracked_point, data


def run_detection_model(frame, confidence):
    global detection_model
    with detection_model_lock:
        if detection_model is None:
            from ultralytics import YOLO
            detection_model = YOLO('yolov8n.pt')
        return detection_model.predict(frame, conf=confidence, verbose=False, device=temporal.inference_device())[0]


def disparity_in_region(disparity_map, box, params):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    inner = (
        int(max(0, left + width * 0.3)),
        int(max(0, top + height * 0.3)),
        int(min(disparity_map.shape[1], right - width * 0.3)),
        int(min(disparity_map.shape[0], bottom - height * 0.3)),
    )
    region = disparity_map[inner[1]:max(inner[1] + 1, inner[3]), inner[0]:max(inner[0] + 1, inner[2])]
    valid = region[np.isfinite(region) & (region > params['minDisparity'])]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def detect_objects(frame_left, disparity_map, active_calibration, params, options):
    result = {'view': options['view'], 'metricAvailable': False, 'objects': [], 'error': None}
    metric = bool(active_calibration and active_calibration.get('metric') and active_calibration.get('q') is not None)
    result['metricAvailable'] = metric
    try:
        prediction = run_detection_model(frame_left, options['confidence'])
        boxes = prediction.boxes
        if boxes is None or boxes.xyxy is None:
            return result
        coordinates = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), class_index, confidence in list(zip(coordinates, classes, confidences))[:20]:
            corners = [pose_disparity_pixel(x, y, active_calibration, False) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
            rect_box = [
                min(point[0] for point in corners), min(point[1] for point in corners),
                max(point[0] for point in corners), max(point[1] for point in corners),
            ]
            disparity = disparity_in_region(disparity_map, rect_box, params)
            distance_m = None
            if metric and disparity is not None:
                center_x = (rect_box[0] + rect_box[2]) / 2
                center_y = (rect_box[1] + rect_box[3]) / 2
                homogeneous = active_calibration['q'] @ np.array([center_x, center_y, disparity, 1.0])
                if abs(homogeneous[3]) >= 1e-9:
                    value = float(np.linalg.norm(homogeneous[:3] / homogeneous[3]) / 1000)
                    if math.isfinite(value) and value > 0:
                        distance_m = round(value, 2)
            result['objects'].append({
                'label': prediction.names.get(int(class_index), str(class_index)),
                'confidence': round(float(confidence), 2),
                'box': [round(float(value), 1) for value in (x1, y1, x2, y2)],
                'rectBox': [round(float(value), 1) for value in rect_box],
                'disparity': None if disparity is None else round(disparity, 2),
                'distanceM': distance_m,
            })
    except Exception as error:
        app.logger.exception('object detection failed')
        result['error'] = str(error)[:240]
    return result


def process_disparity(
    frame_left, frame_right, params, return_map=False, measurement_point=None,
    pose_options=None, live=False,
):
    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))
    pose_frame_left = frame_left
    pose_frame_right = frame_right

    if frame_left.shape[1] <= params['numDisparities'] + params['blockSize']:
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400

    frame_left, frame_right, active_calibration, pose_calibration = rectify_pair(frame_left, frame_right)
    if live:
        with temporal_settings_lock:
            temporal_options = temporal_settings.copy()
        with output_settings_lock:
            view_options = output_settings.copy()
        with detection_settings_lock:
            detection_options = detection_settings.copy()
    else:
        temporal_options = None
        view_options = {'view': 'disparity'}
        detection_options = None

    try:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        temporal_data = None
        if temporal_options and temporal_options['autoCorrect']:
            with temporal_lock:
                frame_right, gray_right = drift_corrector.correct(gray_left, gray_right, frame_right)
                temporal_data = {'drift': drift_corrector.status()}
        disparity_map, stereo_backend = compute_disparity(gray_left, gray_right, params)
        if return_map:
            return disparity_map
        tracked_point = None
        if temporal_options and temporal_options['enabled']:
            disparity_map, tracked_point, stage_data = run_temporal_stage(
                frame_left, gray_left, disparity_map, measurement_point, params, temporal_options
            )
            temporal_data = {**(temporal_data or {}), **stage_data}
            if tracked_point is not None:
                measurement_point = (tracked_point['x'], tracked_point['y'])
        measurement = measurement_error = None
        if measurement_point is not None:
            measurement, measurement_error = estimate_point(
                disparity_map, *measurement_point, active_calibration, params
            )
            if tracked_point is not None:
                with temporal_lock:
                    measurement, measurement_error = measurement_tracker.stabilize(
                        measurement, measurement_error,
                        temporal_options['smoothing'], temporal_options['holdFrames'],
                    )
        pose_data = None
        if pose_options and pose_options['enabled']:
            pose_data = estimate_pose_data(
                pose_frame_left, pose_frame_right, disparity_map, pose_calibration,
                params, pose_options
            )
        detection_data = None
        if detection_options and detection_options['enabled']:
            detection_data = detect_objects(pose_frame_left, disparity_map, pose_calibration, params, detection_options)

        output_image = render_output_view(frame_left, frame_right, disparity_map, params, view_options)
        encoded, buffer = cv2.imencode('.jpg', output_image)
        if not encoded:
            app.logger.error('disparity JPEG encoding failed')
            return jsonify({'error': 'Could not encode the disparity image.'}), 500
    except Exception:
        app.logger.exception('disparity processing failed')
        return jsonify({'error': 'Stereo image processing failed; see container logs.'}), 500

    jpeg = buffer.tobytes()
    if live:
        stereo_output_hub.publish(jpeg)
    headers = {'Cache-Control': 'no-store', 'X-Stereo-Backend': stereo_backend, 'X-Output-View': view_options['view']}
    if measurement_point is not None:
        headers['X-Measurement'] = json.dumps({
            'measurement': measurement,
            'error': measurement_error,
            'point': tracked_point,
        })
    if pose_data is not None:
        headers['X-Pose-Data'] = json.dumps(pose_data, separators=(',', ':'))
    if detection_data is not None:
        headers['X-Detection-Data'] = json.dumps(detection_data, separators=(',', ':'))
    if temporal_data is not None:
        headers['X-Temporal-Data'] = json.dumps(temporal_data, separators=(',', ':'))
    return Response(jpeg, mimetype='image/jpeg', headers=headers)


def validate_pose_settings(data):
    if not isinstance(data, dict):
        raise ValueError('Pose settings must be a JSON object.')
    enabled = data.get('enabled', pose_settings['enabled'])
    if not isinstance(enabled, bool):
        raise ValueError('enabled must be a boolean.')
    view = data.get('view', pose_settings['view'])
    if view not in ('left', 'right', 'composite', 'both'):
        raise ValueError('view must be left, right, composite, or both.')
    confidence = data.get('confidence', pose_settings['confidence'])
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError('confidence must be a number.')
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
        raise ValueError('confidence must be between 0.1 and 0.9.')
    return {'enabled': enabled, 'view': view, 'confidence': confidence}


def configured_camera_slots():
    with camera_settings_lock:
        return {camera['slot'] for camera in camera_settings['cameras']}


def validate_tap_settings(data, allowed_slots=None):
    if not isinstance(data, dict):
        raise ValueError('TAP-Net settings must be a JSON object.')
    enabled = data.get('enabled', tap_settings['enabled'])
    if not isinstance(enabled, bool):
        raise ValueError('enabled must be a boolean.')
    camera_slots = data.get('cameraSlots', tap_settings['cameraSlots'])
    if not isinstance(camera_slots, list):
        raise ValueError('cameraSlots must be a list.')
    if allowed_slots is None:
        allowed_slots = configured_camera_slots()
    if any(type(slot) is not int or slot not in allowed_slots for slot in camera_slots):
        raise ValueError('cameraSlots may contain only configured camera slots.')
    if len(set(camera_slots)) != len(camera_slots):
        raise ValueError('cameraSlots must not contain duplicates.')
    composite = data.get('composite', tap_settings['composite'])
    if not isinstance(composite, bool):
        raise ValueError('composite must be a boolean.')
    if enabled and not camera_slots and not composite:
        raise ValueError('Select at least one camera or the composite view.')
    backend = data.get('backend', tap_settings['backend'])
    if backend not in temporal.BACKENDS:
        raise ValueError('backend must be auto, tapir, or lucas-kanade.')
    max_points = integer_setting(data, 'maxPoints', tap_settings['maxPoints'], 8, 200)
    hold_frames = integer_setting(data, 'holdFrames', tap_settings['holdFrames'], 0, 60)
    return {
        'enabled': enabled,
        'cameraSlots': sorted(camera_slots),
        'composite': composite,
        'backend': backend,
        'maxPoints': max_points,
        'holdFrames': hold_frames,
    }


def integer_setting(data, key, fallback, minimum, maximum):
    value = data.get(key, fallback)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{key} must be an integer from {minimum} to {maximum}.')
    return value


def number_setting(data, key, fallback, minimum, maximum):
    value = data.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{key} must be a number.')
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{key} must be between {minimum} and {maximum}.')
    return value


def boolean_setting(data, key, fallback):
    value = data.get(key, fallback)
    if not isinstance(value, bool):
        raise ValueError(f'{key} must be a boolean.')
    return value


def validate_output_settings(data):
    if not isinstance(data, dict):
        raise ValueError('Output settings must be a JSON object.')
    view = data.get('view', output_settings['view'])
    if view not in stereo_extras.OUTPUT_VIEWS:
        raise ValueError('view must be disparity, anaglyph, or bokeh.')
    blur = integer_setting(data, 'bokehBlur', output_settings['bokehBlur'], 5, 75)
    if blur % 2 == 0:
        raise ValueError('bokehBlur must be odd.')
    return {
        'view': view,
        'bokehThreshold': number_setting(data, 'bokehThreshold', output_settings['bokehThreshold'], 0.05, 0.95),
        'bokehBlur': blur,
    }


def validate_detection_settings(data):
    if not isinstance(data, dict):
        raise ValueError('Detection settings must be a JSON object.')
    view = data.get('view', detection_settings['view'])
    if view not in ('left', 'composite', 'both'):
        raise ValueError('view must be left, composite, or both.')
    return {
        'enabled': boolean_setting(data, 'enabled', detection_settings['enabled']),
        'view': view,
        'confidence': number_setting(data, 'confidence', detection_settings['confidence'], 0.1, 0.9),
    }


def validate_temporal_settings(data):
    if not isinstance(data, dict):
        raise ValueError('Temporal settings must be a JSON object.')
    backend = data.get('backend', temporal_settings['backend'])
    if backend not in temporal.BACKENDS:
        raise ValueError('backend must be auto, tapir, or lucas-kanade.')
    return {
        'enabled': boolean_setting(data, 'enabled', temporal_settings['enabled']),
        'smoothing': number_setting(data, 'smoothing', temporal_settings['smoothing'], 0.0, 0.95),
        'holdFrames': integer_setting(data, 'holdFrames', temporal_settings['holdFrames'], 0, 60),
        'trackMeasurement': boolean_setting(data, 'trackMeasurement', temporal_settings['trackMeasurement']),
        'autoCorrect': boolean_setting(data, 'autoCorrect', temporal_settings['autoCorrect']),
        'backend': backend,
    }


def run_pose_model(frame, confidence, tracking):
    global pose_model
    with pose_model_lock:
        if pose_model is None:
            from ultralytics import YOLO
            pose_model = YOLO('yolov8n-pose.pt')
        device = temporal.inference_device()
        if tracking:
            return pose_model.track(
                frame, persist=True, conf=confidence, verbose=False, device=device
            )[0]
        return pose_model.predict(frame, conf=confidence, verbose=False, device=device)[0]


def pose_disparity_pixel(x, y, active_calibration, right_view):
    if not active_calibration:
        return float(x), float(y)
    camera = 'right' if right_view else 'left'
    if active_calibration.get('metric') and f'camera_{camera}' in active_calibration:
        point = np.array([[[x, y]]], dtype=np.float32)
        rectified = cv2.undistortPoints(
            point,
            active_calibration[f'camera_{camera}'],
            active_calibration[f'distortion_{camera}'],
            R=active_calibration[f'rect_{camera}'],
            P=active_calibration[f'projection_{camera}'],
        )
        return float(rectified[0, 0, 0]), float(rectified[0, 0, 1])
    transform = active_calibration.get(f'point_transform_{camera}')
    if transform is not None:
        point = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), transform
        )
        return float(point[0, 0, 0]), float(point[0, 0, 1])
    return float(x), float(y)


def pose_people(frame, tracking, confidence, disparity_map=None, active_calibration=None, params=None, right_view=False):
    result = run_pose_model(frame, confidence, tracking)
    if result.keypoints is None or result.keypoints.xy is None:
        return []
    coordinates = result.keypoints.xy.cpu().numpy()
    confidences = result.keypoints.conf
    confidences = confidences.cpu().numpy() if confidences is not None else None
    track_ids = result.boxes.id
    track_ids = track_ids.cpu().tolist() if track_ids is not None else [None] * len(coordinates)
    people = []
    for person_index, joints in enumerate(coordinates[:6]):
        points = []
        for joint_index, (x_value, y_value) in enumerate(joints):
            disparity_x, disparity_y = pose_disparity_pixel(
                x_value, y_value, active_calibration, right_view
            )
            point = {
                'index': joint_index,
                'x': round(float(x_value), 1),
                'y': round(float(y_value), 1),
                'disparityX': round(disparity_x, 1),
                'disparityY': round(disparity_y, 1),
                'confidence': round(float(confidences[person_index][joint_index]), 2)
                if confidences is not None else 1,
                'xyzM': None,
                'distanceM': None,
            }
            if disparity_map is not None and active_calibration and active_calibration.get('metric'):
                x, y = int(round(disparity_x)), int(round(disparity_y))
                if right_view:
                    x = min(disparity_map.shape[1] - 1, x + params['numDisparities'] // 2)
                if 0 <= x < disparity_map.shape[1] and 0 <= y < disparity_map.shape[0]:
                    region = disparity_map[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
                    valid = region[np.isfinite(region) & (region > params['minDisparity'])]
                    if valid.size >= 3:
                        disparity = float(np.median(valid))
                        if right_view:
                            x = min(disparity_map.shape[1] - 1, int(round(disparity_x + disparity)))
                        homogeneous = active_calibration['q'] @ np.array([x, y, disparity, 1.0])
                        if abs(homogeneous[3]) >= 1e-9:
                            xyz_m = homogeneous[:3] / homogeneous[3] / 1000
                            distance_m = float(np.linalg.norm(xyz_m))
                            if np.isfinite(xyz_m).all() and math.isfinite(distance_m) and distance_m > 0:
                                point['xyzM'] = [round(float(value), 3) for value in xyz_m]
                                point['distanceM'] = round(distance_m, 3)
            points.append(point)
        people.append({'trackId': track_ids[person_index], 'points': points})
    return people


def estimate_pose_data(frame_left, frame_right, disparity_map, active_calibration, params, options):
    metric_available = bool(
        active_calibration and active_calibration.get('metric') and active_calibration.get('q') is not None
    )
    result = {
        'view': options['view'],
        'metricAvailable': metric_available,
        'left': [],
        'right': [],
        'error': None,
    }
    try:
        if options['view'] in ('left', 'composite', 'both'):
            result['left'] = pose_people(
                frame_left, True, options['confidence'], disparity_map,
                active_calibration, params
            )
        if options['view'] in ('right', 'both'):
            result['right'] = pose_people(
                frame_right, False, options['confidence'], disparity_map,
                active_calibration, params, right_view=True
            )
    except Exception as error:
        app.logger.exception('pose estimation failed')
        result['error'] = str(error)[:240]
    return result


def calibration_status():
    with calibration_lock:
        active = calibration_state['active']
        result = {
            'captures': len(calibration_state['captures']),
            'requiredCaptures': CALIBRATION_VIEWS_REQUIRED,
            'calibrated': active is not None,
            'method': calibration_state['method'],
            'metric': bool(active and active.get('metric', True)),
        }
        if active:
            result.update({
                'imageSize': list(active['image_size']),
            })
            if active.get('metric', True):
                result.update({
                    'baselineMm': round(active['baseline_mm'], 2),
                    'focalLengthPx': round(active['focal_length_px'], 2),
                    'squareSizeMm': active['square_size_mm'],
                    'columns': active['columns'],
                    'rows': active['rows'],
                })
            else:
                result.update({
                    'matchedFeatures': active['matched_features'],
                    'inliers': active['inliers'],
                })
        return result


def build_feature_rectification(frame_left, frame_right):
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_left = clahe.apply(gray_left)
    enhanced_right = clahe.apply(gray_right)
    detector = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    keypoints_left, descriptors_left = detector.detectAndCompute(enhanced_left, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(enhanced_right, None)
    if descriptors_left is None or descriptors_right is None:
        raise ValueError('Not enough image texture found. Improve the lighting or aim both cameras at a detailed scene.')

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_left, descriptors_right, k=2)
    matches = [
        neighbors[0] for neighbors in pairs
        if len(neighbors) == 2 and neighbors[0].distance < 0.75 * neighbors[1].distance
    ]
    if len(matches) < 20:
        raise ValueError('Too few reliable features match between the cameras. Aim both cameras at a shared, detailed scene.')

    points_left = np.float32([keypoints_left[match.queryIdx].pt for match in matches])
    points_right = np.float32([keypoints_right[match.trainIdx].pt for match in matches])
    height, width = frame_left.shape[:2]
    fundamental, mask = cv2.findFundamentalMat(
        points_left,
        points_right,
        cv2.FM_RANSAC,
        max(1.0, width * 0.0015),
        0.999,
    )
    if fundamental is None or mask is None:
        raise ValueError('Could not estimate camera alignment. Try a brighter, more detailed shared scene.')
    inliers = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 15 or inlier_count / len(matches) < 0.25:
        raise ValueError('Feature matches were inconsistent. Keep both cameras fixed and capture a shared, detailed scene.')

    rectified, homography_left, homography_right = cv2.stereoRectifyUncalibrated(
        points_left[inliers],
        points_right[inliers],
        fundamental,
        (width, height),
    )
    if not rectified:
        raise ValueError('Could not rectify the camera pair from these feature matches.')

    maps = []
    for homography in (homography_left, homography_right):
        if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
            raise ValueError('The estimated camera alignment is unstable. Capture another shared scene.')
        map_x, map_y = cv2.initUndistortRectifyMap(
            np.eye(3), np.zeros(5), homography, np.eye(3), (width, height), cv2.CV_32FC1
        )
        maps.append((map_x, map_y))

    return {
        'image_size': (width, height),
        'map_left': maps[0],
        'map_right': maps[1],
        'point_transform_left': homography_left,
        'point_transform_right': homography_right,
        'q': None,
        'metric': False,
        'matched_features': len(matches),
        'inliers': inlier_count,
    }


def build_stereo_calibration(captures, board, image_size):
    columns, rows, square_size_mm = board
    object_points = np.zeros((rows * columns, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_mm
    object_sets = [object_points.copy() for _ in captures]
    left_sets = [capture[0] for capture in captures]
    right_sets = [capture[1] for capture in captures]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    _, camera_left, distortion_left, _, _ = cv2.calibrateCamera(
        object_sets, left_sets, image_size, None, None
    )
    _, camera_right, distortion_right, _, _ = cv2.calibrateCamera(
        object_sets, right_sets, image_size, None, None
    )
    _, camera_left, distortion_left, camera_right, distortion_right, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_sets,
        left_sets,
        right_sets,
        camera_left,
        distortion_left,
        camera_right,
        distortion_right,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rect_left, rect_right, projection_left, projection_right, reprojection, _, _ = cv2.stereoRectify(
        camera_left,
        distortion_left,
        camera_right,
        distortion_right,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map_left = cv2.initUndistortRectifyMap(
        camera_left, distortion_left, rect_left, projection_left, image_size, cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        camera_right, distortion_right, rect_right, projection_right, image_size, cv2.CV_32FC1
    )
    return {
        'image_size': image_size,
        'map_left': map_left,
        'map_right': map_right,
        'camera_left': camera_left,
        'distortion_left': distortion_left,
        'rect_left': rect_left,
        'projection_left': projection_left,
        'camera_right': camera_right,
        'distortion_right': distortion_right,
        'rect_right': rect_right,
        'projection_right': projection_right,
        'q': reprojection,
        'baseline_mm': float(np.linalg.norm(translation)),
        'focal_length_px': float(projection_left[0, 0]),
        'square_size_mm': square_size_mm,
        'columns': columns,
        'rows': rows,
    }


def find_calibration_corners(gray, pattern_size, roi=None):
    search_regions = []
    if roi:
        x, y, region_width, region_height = roi
        search_regions.append((gray[y:y + region_height, x:x + region_width], x, y))
    search_regions.append((gray, 0, 0))

    for region, offset_x, offset_y in search_regions:
        if region.size == 0:
            continue
        normalized = cv2.equalizeHist(region)
        if hasattr(cv2, 'findChessboardCornersSB'):
            sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            try:
                found, corners = cv2.findChessboardCornersSB(normalized, pattern_size, sb_flags)
            except cv2.error:
                found, corners = False, None
            if found:
                corners[:, 0, 0] += offset_x
                corners[:, 0, 1] += offset_y
                return corners

        classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        for candidate in (region, normalized):
            try:
                found, corners = cv2.findChessboardCorners(candidate, pattern_size, classic_flags)
            except cv2.error:
                continue
            if found:
                corners[:, 0, 0] += offset_x
                corners[:, 0, 1] += offset_y
                return corners
    return None


@app.route('/api/stereo-calibration', methods=['GET', 'POST', 'DELETE'])
def stereo_calibration():
    if request.method == 'GET':
        return jsonify(calibration_status())
    if request.method == 'DELETE':
        with calibration_lock:
            calibration_state.update({'captures': [], 'board': None, 'active': None, 'method': None})
        return jsonify(calibration_status())

    frame_left = decode_frame(request.files.get('left'))
    frame_right = decode_frame(request.files.get('right'))
    if frame_left is None or frame_right is None:
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400
    if frame_left.shape[:2] != frame_right.shape[:2]:
        return jsonify({'error': 'Both camera frames must have the same resolution.'}), 400
    method = request.form.get('method', 'checkerboard')
    if method not in ('checkerboard', 'feature'):
        return jsonify({'error': 'Calibration method must be checkerboard or feature.'}), 400
    if method == 'feature':
        try:
            active = build_feature_rectification(frame_left, frame_right)
        except (cv2.error, ValueError) as error:
            return jsonify({'error': str(error) or 'Feature-based calibration failed.'}), 422
        with calibration_lock:
            calibration_state.update({
                'captures': [], 'board': None, 'active': active, 'method': 'feature'
            })
        return jsonify(calibration_status())

    try:
        columns = int(request.form.get('columns', '9'))
        rows = int(request.form.get('rows', '6'))
        square_size_mm = float(request.form.get('squareSizeMm', '25'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Board dimensions and square size must be numeric.'}), 400
    if not (4 <= columns <= 15 and 4 <= rows <= 12 and math.isfinite(square_size_mm) and 5 <= square_size_mm <= 100):
        return jsonify({'error': 'Use 4-15 columns, 4-12 rows, and a square size from 5-100 mm.'}), 400
    if frame_left.shape[:2] != frame_right.shape[:2]:
        return jsonify({'error': 'Both camera frames must have the same resolution.'}), 400

    try:
        raw_rois = json.loads(request.form.get('rois', '{}'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Board location hints must be valid JSON.'}), 400
    if not isinstance(raw_rois, dict):
        return jsonify({'error': 'Board location hints must be an object.'}), 400
    rois = {}
    for camera in ('left', 'right'):
        roi = raw_rois.get(camera)
        if roi is None:
            continue
        if not isinstance(roi, dict):
            return jsonify({'error': f'The {camera} board location hint is invalid.'}), 400
        try:
            x, y, roi_width, roi_height = (float(roi[key]) for key in ('x', 'y', 'width', 'height'))
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': f'The {camera} board location hint is invalid.'}), 400
        if (not all(math.isfinite(value) for value in (x, y, roi_width, roi_height))
                or x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0
                or x + roi_width > 1 or y + roi_height > 1):
            return jsonify({'error': f'The {camera} board location hint is outside its image.'}), 400
        image_height, image_width = frame_left.shape[:2]
        left = max(0, round(x * image_width))
        top = max(0, round(y * image_height))
        right = min(image_width, round((x + roi_width) * image_width))
        bottom = min(image_height, round((y + roi_height) * image_height))
        rois[camera] = (left, top, right - left, bottom - top)

    board = (columns, rows, square_size_mm)
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    pattern_size = (columns, rows)
    corners_left = find_calibration_corners(gray_left, pattern_size, rois.get('left'))
    corners_right = find_calibration_corners(gray_right, pattern_size, rois.get('right'))
    missing = [camera for camera, corners in (('left', corners_left), ('right', corners_right)) if corners is None]
    if missing:
        labels = ' and '.join(f'{camera} camera' for camera in missing)
        return jsonify({
            'error': f'Chessboard not found in the {labels} view. Drag a box around the full board in that preview, then retry.',
            'missingViews': missing,
        }), 422
    refine = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), refine)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), refine)
    image_size = (frame_left.shape[1], frame_left.shape[0])
    with calibration_lock:
        if (calibration_state['method'] != 'checkerboard'
                or calibration_state['board'] != board
                or calibration_state['captures'] and calibration_state['captures'][0][2] != image_size):
            calibration_state.update({'captures': [], 'board': board, 'active': None})
        calibration_state['board'] = board
        calibration_state['method'] = 'checkerboard'
        calibration_state['captures'].append((corners_left, corners_right, image_size))
        captures = calibration_state['captures'][:]

    calibrated = False
    calibration_error = None
    if len(captures) >= CALIBRATION_VIEWS_REQUIRED:
        try:
            active = build_stereo_calibration(
                [(left, right) for left, right, _ in captures], board, image_size
            )
            with calibration_lock:
                calibration_state['active'] = active
            calibrated = True
        except cv2.error as error:
            calibration_error = str(error).splitlines()[0]
            app.logger.warning('stereo calibration failed: %s', calibration_error)

    status = calibration_status()
    status['accepted'] = True
    status['calibrated'] = calibrated or status['calibrated']
    if calibration_error:
        status['calibrationError'] = calibration_error
    return jsonify(status)


@app.route('/api/measure', methods=['POST'])
def measure_point():
    frame_left = decode_frame(request.files.get('left'))
    frame_right = decode_frame(request.files.get('right'))
    if frame_left is None or frame_right is None:
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400
    try:
        x = int(request.form['x'])
        y = int(request.form['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Select a point on the disparity map.'}), 400
    with calibration_lock:
        active = calibration_state['active']
    if not active or active['image_size'] != (frame_left.shape[1], frame_left.shape[0]):
        return jsonify({'error': 'Calibrate this camera resolution before measuring distance.'}), 409
    with params_lock:
        params = stereo_params.copy()
    disparity_map = process_disparity(frame_left, frame_right, params, return_map=True)
    if isinstance(disparity_map, tuple):
        return disparity_map
    if x < 0 or y < 0 or x >= disparity_map.shape[1] or y >= disparity_map.shape[0]:
        return jsonify({'error': 'Selected point is outside the disparity image.'}), 400
    measurement, error = estimate_point(disparity_map, x, y, active, params)
    if error:
        return jsonify({'error': error}), 422
    return jsonify(measurement)


def calibration_target_parameters():
    try:
        columns = int(request.args.get('columns', '9'))
        rows = int(request.args.get('rows', '6'))
        square_size_mm = float(request.args.get('squareSizeMm', '18'))
    except ValueError:
        raise ValueError('Target dimensions must be numeric.') from None
    if not (4 <= columns <= 15 and 4 <= rows <= 12 and math.isfinite(square_size_mm) and 5 <= square_size_mm <= 100):
        raise ValueError('Target dimensions are outside supported bounds.')
    return columns, rows, square_size_mm


@app.route('/calibration-target.svg')
def calibration_target():
    try:
        columns, rows, square_size_mm = calibration_target_parameters()
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    square_columns, square_rows = columns + 1, rows + 1
    width_mm, height_mm = square_columns * square_size_mm, square_rows * square_size_mm
    squares = ''.join(
        f'<rect x="{column * square_size_mm}" y="{row * square_size_mm}" width="{square_size_mm}" height="{square_size_mm}" fill="#111"/>'
        for row in range(square_rows) for column in range(square_columns)
        if (row + column) % 2 == 0
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}"><rect width="100%" height="100%" fill="white"/>{squares}</svg>'
    )
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-store'})


@app.route('/calibration-target.pdf')
def calibration_target_pdf():
    try:
        columns, rows, square_size_mm = calibration_target_parameters()
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    square_columns, square_rows = columns + 1, rows + 1
    page_width = square_columns * square_size_mm * mm
    page_height = square_rows * square_size_mm * mm
    output = BytesIO()
    document = canvas.Canvas(output, pagesize=(page_width, page_height), pageCompression=1)
    document.setTitle('Stereo calibration checkerboard')
    document.setFillColorRGB(1, 1, 1)
    document.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    document.setFillColorRGB(0, 0, 0)
    square = square_size_mm * mm
    for row in range(square_rows):
        for column in range(square_columns):
            if (row + column) % 2 == 0:
                document.rect(column * square, (square_rows - row - 1) * square, square, square, fill=1, stroke=0)
    document.showPage()
    document.save()
    output.seek(0)
    return send_file(
        output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='stereo-calibration-target.pdf',
        max_age=0,
    )


@app.route('/api/client-log', methods=['POST'])
def client_log():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON diagnostic event.'}), 400
    event = data.get('event')
    details = data.get('details', {})
    if not isinstance(event, str) or not isinstance(details, dict):
        return jsonify({'error': 'Diagnostic event and details have invalid types.'}), 400
    details_json = json.dumps(details, ensure_ascii=True)
    if len(details_json) > 12000:
        return jsonify({'error': 'Diagnostic details are too large.'}), 413
    app.logger.info(
        'browser_camera event=%s details=%s',
        event[:80],
        details_json
    )
    return jsonify({'status': 'logged'})

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/multi-camera')
def multi_camera():
    return render_template('multi_camera.html')


@app.route('/multy-camera')
def multi_camera_spelling_alias():
    return redirect(url_for('multi_camera'), code=302)


@app.route('/api/pose-settings', methods=['GET', 'POST'])
def pose_settings_endpoint():
    global pose_settings
    if request.method == 'GET':
        with pose_settings_lock:
            return jsonify(pose_settings.copy())
    data = request.get_json(silent=True)
    try:
        updated = validate_pose_settings(data)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    with pose_settings_lock:
        pose_settings = updated
    return jsonify(pose_settings.copy())


@app.route('/api/tap-settings', methods=['GET', 'POST'])
def tap_settings_endpoint():
    global tap_settings
    if request.method == 'GET':
        with tap_settings_lock:
            settings = tap_settings.copy()
    else:
        data = request.get_json(silent=True)
        try:
            updated = validate_tap_settings(data)
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        with tap_settings_lock:
            tap_settings = updated
            settings = tap_settings.copy()
        tap_runtime.reset()
    available, message = temporal.tapir_availability()
    return jsonify({
        **settings,
        'available': True,
        'tapirAvailable': available,
        'tapirMessage': message,
        'status': 'tracking' if settings['enabled'] else 'disabled',
    })


@app.route('/api/tap/output')
def tap_output_endpoint():
    client_id = request.args.get('clientId', '')
    source = request.args.get('source', 'browser')
    if not valid_camera_client_id(client_id):
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    if source not in ('browser', 'docker'):
        return jsonify({'error': 'source must be browser or docker.'}), 400

    def generate():
        while True:
            started = time.perf_counter()
            with tap_settings_lock:
                settings = tap_settings.copy()
            frames = multiview_runtime.prepare_frames(multiview_frames_for(source, client_id))
            try:
                image, status = tap_runtime.render(frames, settings)
                jpeg = multiview.encode_jpeg(image)
            except Exception:
                app.logger.exception('temporal tracking output failed')
                jpeg = multiview.encode_jpeg(multiview.message_image('Tracking failed. See the application log.'))
                status = {'ok': False, 'message': 'Tracking failed.', 'targets': []}
            tap_runtime.remember_status(client_id, status)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(max(0.05, 0.15 - (time.perf_counter() - started)))

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


@app.route('/api/tap/status')
def tap_status_endpoint():
    return jsonify(tap_runtime.status_for(request.args.get('clientId', '')))


def settings_endpoint(lock, getter, setter, validator, on_change=None):
    if request.method == 'GET':
        with lock:
            return jsonify(getter())
    try:
        updated = validator(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    with lock:
        changed = updated != getter()
        setter(updated)
    if changed and on_change:
        on_change()
    with lock:
        return jsonify(getter())


@app.route('/api/output-settings', methods=['GET', 'POST'])
def output_settings_endpoint():
    def setter(value):
        global output_settings
        output_settings = value
    return settings_endpoint(output_settings_lock, lambda: output_settings.copy(), setter, validate_output_settings)


@app.route('/api/detection-settings', methods=['GET', 'POST'])
def detection_settings_endpoint():
    def setter(value):
        global detection_settings
        detection_settings = value
    return settings_endpoint(detection_settings_lock, lambda: detection_settings.copy(), setter, validate_detection_settings)


def reset_temporal_state():
    with temporal_lock:
        disparity_filter.reset()
        measurement_tracker.reset()
        drift_corrector.reset()


@app.route('/api/temporal-settings', methods=['GET', 'POST'])
def temporal_settings_endpoint():
    def setter(value):
        global temporal_settings
        temporal_settings = value

    def getter():
        available, message = temporal.tapir_availability()
        with temporal_lock:
            drift = drift_corrector.status()
            backend = measurement_tracker.backend.name
        return {
            **temporal_settings,
            'activeBackend': backend,
            'tapirAvailable': available,
            'tapirMessage': message,
            'drift': drift,
        }

    if request.method == 'POST':
        try:
            updated = validate_temporal_settings(request.get_json(silent=True))
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        with temporal_settings_lock:
            changed = updated != temporal_settings
            setter(updated)
        if changed:
            reset_temporal_state()
    with temporal_settings_lock:
        return jsonify(getter())


@app.route('/api/acceleration')
def acceleration_endpoint():
    with acceleration_lock:
        state = acceleration_state.copy()
    return jsonify({
        'opencvCudaDevices': opencv_cuda_devices(),
        'cudaStereoAvailable': CUDA_STEREO_AVAILABLE,
        'cudaStereoDisparities': list(CUDA_SGM_DISPARITIES),
        'torchCuda': torch_cuda_available(),
        'inferenceDevice': temporal.inference_device(),
        'open3d': stereo_extras.open3d_available(),
        **state,
    })


@app.route('/api/stereo/stream')
def stereo_stream_endpoint():
    idle = multiview.encode_jpeg(multiview.message_image('Waiting for the stereo monitor to process frames.'))
    return Response(
        stereo_output_hub.frames(idle),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


@app.route('/api/pointcloud', methods=['POST'])
def point_cloud_endpoint():
    frame_left = decode_frame(request.files.get('left'))
    frame_right = decode_frame(request.files.get('right'))
    if frame_left is None or frame_right is None:
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400
    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))
    output_format = request.form.get('format', 'points')
    mesh_method = request.form.get('meshMethod', 'grid')
    if output_format not in ('points', 'mesh'):
        return jsonify({'error': 'format must be points or mesh.'}), 400
    if mesh_method not in stereo_extras.MESH_METHODS:
        return jsonify({'error': 'meshMethod must be grid or poisson.'}), 400
    with params_lock:
        params = stereo_params.copy()
    rect_left, rect_right, _, calibration = rectify_pair(frame_left, frame_right)
    if not calibration or not calibration.get('metric') or calibration.get('q') is None:
        return jsonify({'error': 'Checkerboard calibration at this camera resolution is required for 3D export.'}), 409
    if rect_left.shape[1] <= params['numDisparities'] + params['blockSize']:
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400
    try:
        disparity_map, _ = compute_disparity(
            cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY), params
        )
        points, colors, mask = stereo_extras.reproject(disparity_map, calibration['q'], rect_left, params['minDisparity'])
        if np.count_nonzero(mask) < 100:
            return jsonify({'error': 'Too few valid depth points. Improve lighting and scene texture, then retry.'}), 422
        if output_format == 'points':
            payload = stereo_extras.ply_point_cloud(points[mask], colors[mask])
            name = 'stereo-point-cloud.ply'
        elif mesh_method == 'poisson':
            try:
                payload = stereo_extras.ply_mesh(*stereo_extras.poisson_mesh(points[mask], colors[mask]))
            except ImportError:
                return jsonify({'error': 'Poisson reconstruction needs Open3D. Build with INSTALL_OPEN3D=1 or use the grid mesh.'}), 501
            name = 'stereo-poisson-mesh.ply'
        else:
            payload = stereo_extras.ply_mesh(*stereo_extras.grid_mesh(points, colors, mask))
            name = 'stereo-mesh.ply'
    except Exception:
        app.logger.exception('3D export failed')
        return jsonify({'error': '3D export failed; see container logs.'}), 500
    return send_file(
        BytesIO(payload),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=name,
        max_age=0,
    )


@app.route('/api/camera-settings', methods=['GET', 'POST'])
def camera_settings_endpoint():
    global camera_settings
    if request.method == 'GET':
        with camera_settings_lock:
            return jsonify(camera_settings)

    data = request.get_json(silent=True)
    try:
        updated = validate_camera_settings(data)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    with camera_settings_lock:
        changed = updated != camera_settings
        if changed or not CAMERA_SETTINGS_PATH.is_file():
            try:
                save_camera_settings(updated)
            except OSError as error:
                return jsonify({'error': f'Could not persist camera settings: {error}'}), 500
        if changed:
            camera_settings = updated
            camera_stream_manager.configure(camera_settings)
        return jsonify(camera_settings)


@app.route('/api/cameras/available')
def available_cameras_endpoint():
    return jsonify({
        'devices': [
            {
                'deviceIndex': device_index,
                'path': f'/dev/video{device_index}',
            }
            for device_index in AVAILABLE_CAMERA_INDICES
        ]
    })


@app.route('/api/cameras/start', methods=['POST'])
def start_cameras_endpoint():
    data = request.get_json(silent=True)
    client_id = data.get('clientId') if isinstance(data, dict) else None
    if not isinstance(client_id, str) or not 8 <= len(client_id) <= 100:
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    with camera_settings_lock:
        active_cameras = [camera for camera in camera_settings['cameras'] if camera['active']]
        if not active_cameras:
            return jsonify({'error': 'Enable at least one camera before connecting.'}), 400
        camera_stream_manager.start(camera_settings, client_id)
    return jsonify({'status': 'starting', 'cameras': camera_stream_manager.status()})


@app.route('/api/cameras/stop', methods=['POST'])
def stop_cameras_endpoint():
    data = request.get_json(silent=True)
    client_id = data.get('clientId') if isinstance(data, dict) else None
    if not isinstance(client_id, str) or not 8 <= len(client_id) <= 100:
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    camera_stream_manager.stop(client_id)
    return jsonify({'status': 'stopped'})


@app.route('/api/cameras/status', methods=['GET'])
def camera_status_endpoint():
    return jsonify({'cameras': camera_stream_manager.status()})


@app.route('/api/cameras/<int:slot>/stream')
def camera_stream_endpoint(slot):
    stream = camera_stream_manager.stream(slot)
    if stream is None:
        return jsonify({'error': 'This camera slot is not active.'}), 404
    return Response(
        stream.frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


@app.route('/api/disparity', methods=['POST'])
@app.route('/api/preview', methods=['POST'])
def disparity():
    is_preview = request.path == '/api/preview'
    started = time.perf_counter()
    left_upload = request.files.get('left')
    right_upload = request.files.get('right')
    app.logger.info(
        'disparity request content_length=%s left_present=%s right_present=%s',
        request.content_length,
        left_upload is not None,
        right_upload is not None
    )
    frame_left = decode_frame(left_upload)
    frame_right = decode_frame(right_upload)
    if frame_left is None or frame_right is None:
        app.logger.warning(
            'disparity rejected invalid image left_decoded=%s right_decoded=%s',
            frame_left is not None,
            frame_right is not None
        )
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400

    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))

    with params_lock:
        current_params = stereo_params.copy()
    if is_preview:
        try:
            data = json.loads(request.form.get('params', '{}'))
            params = validate_calibration_params(data, current_params)
        except (json.JSONDecodeError, ValueError) as error:
            message = str(error) or 'Calibration settings must be a JSON object.'
            return jsonify({'error': message}), 400
    else:
        params = current_params
    with pose_settings_lock:
        current_pose_settings = pose_settings.copy()

    measurement_point = None
    if not is_preview and ('measureX' in request.form or 'measureY' in request.form):
        try:
            measurement_point = (int(request.form['measureX']), int(request.form['measureY']))
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'Measurement coordinates must be integers.'}), 400
    result = process_disparity(
        frame_left, frame_right, params,
        measurement_point=measurement_point,
        pose_options=None if is_preview else current_pose_settings,
        live=not is_preview,
    )
    if isinstance(result, tuple):
        return result

    app.logger.info(
        '%s success left_shape=%s right_shape=%s params=%s elapsed_ms=%.1f',
        'preview' if is_preview else 'disparity',
        frame_left.shape,
        frame_right.shape,
        params,
        (time.perf_counter() - started) * 1000
    )
    return result

@app.route('/api/calibrate', methods=['GET', 'POST'])
def calibrate():
    if request.method == 'GET':
        with params_lock:
            params = stereo_params.copy()
        return jsonify({'params': params})
    data = request.get_json(silent=True) or {}
    with params_lock:
        current_params = stereo_params.copy()
    try:
        params = validate_calibration_params(data, current_params)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    with params_lock:
        stereo_params.update(params)
        params = stereo_params.copy()
    app.logger.info('stereo parameters updated params=%s', params)
    return jsonify({'status': 'success', 'params': params})

def valid_camera_client_id(client_id):
    return isinstance(client_id, str) and 8 <= len(client_id) <= 100


def multiview_frames_for(source, client_id):
    if source == 'docker':
        return camera_stream_manager.latest_frames()
    return multiview_runtime.hub.snapshot(client_id)


def store_uploaded_frames(client_id):
    frames = {}
    for key, upload in request.files.items():
        if not key.startswith('camera') or not key[6:].isdigit():
            continue
        slot = int(key[6:])
        if slot not in range(1, 17):
            continue
        frame = decode_frame(upload)
        if frame is not None:
            frames[slot] = frame
    if frames:
        multiview_runtime.hub.update(client_id, frames)
    return frames


@app.route('/api/multiview/settings', methods=['GET', 'POST'])
def multiview_settings_endpoint():
    if request.method == 'GET':
        return jsonify(multiview_runtime.current_settings())
    data = request.get_json(silent=True)
    try:
        updated = multiview_runtime.update_settings(data or {})
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    return jsonify(updated)


@app.route('/api/multiview/frames', methods=['POST'])
def multiview_frames_endpoint():
    client_id = request.form.get('clientId')
    if not valid_camera_client_id(client_id):
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    frames = store_uploaded_frames(client_id)
    if not frames:
        return jsonify({'error': 'Upload a valid image for at least one camera.'}), 400
    return jsonify({'slots': sorted(frames)})


@app.route('/api/multiview/status')
def multiview_status_endpoint():
    client_id = request.args.get('clientId', '')
    return jsonify(multiview_runtime.status_for(client_id))


@app.route('/api/multiview/output')
def multiview_output_endpoint():
    client_id = request.args.get('clientId', '')
    source = request.args.get('source', 'browser')
    if not valid_camera_client_id(client_id):
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    if source not in ('browser', 'docker'):
        return jsonify({'error': 'source must be browser or docker.'}), 400

    def generate():
        while True:
            frames = multiview_frames_for(source, client_id)
            with params_lock:
                current_params = stereo_params.copy()
            try:
                jpeg, status = multiview_runtime.render(
                    frames,
                    current_params,
                    COLOR_MAPS[current_params['colorMap']],
                    STEREO_MODES[current_params['mode']],
                )
            except Exception:
                app.logger.exception('multiview processing failed')
                jpeg = multiview.encode_jpeg(multiview.message_image('Processing failed. See the application log.'))
                status = {'mode': multiview_runtime.current_settings()['mode'], 'ok': False, 'message': 'Processing failed.'}
            multiview_runtime.remember_status(client_id, status)
            yield (
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + jpeg
                + b'\r\n'
            )
            time.sleep(0.2)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


@app.route('/api/charuco', methods=['GET', 'DELETE'])
def charuco_endpoint():
    if request.method == 'DELETE':
        return jsonify(multiview_runtime.reset_charuco())
    return jsonify(multiview_runtime.charuco_status())


@app.route('/api/charuco/settings', methods=['POST'])
def charuco_settings_endpoint():
    data = request.get_json(silent=True)
    try:
        status = multiview_runtime.update_board(data or {})
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    return jsonify(status)


@app.route('/api/charuco/capture', methods=['POST'])
def charuco_capture_endpoint():
    if request.files:
        client_id = request.form.get('clientId')
        source = request.form.get('source', 'browser')
    else:
        data = request.get_json(silent=True) or {}
        client_id = data.get('clientId')
        source = data.get('source', 'docker')
    if not valid_camera_client_id(client_id):
        return jsonify({'error': 'A valid camera client ID is required.'}), 400
    if source not in ('browser', 'docker'):
        return jsonify({'error': 'source must be browser or docker.'}), 400
    if source == 'browser':
        frames = store_uploaded_frames(client_id)
    else:
        frames = camera_stream_manager.latest_frames()
    try:
        status = multiview_runtime.capture(frames)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    return jsonify(status)


@app.route('/api/charuco/calibrate', methods=['POST'])
def charuco_calibrate_endpoint():
    try:
        status = multiview_runtime.calibrate()
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    return jsonify(status)


@app.route('/charuco-board.png')
def charuco_board_png():
    encoded, buffer = cv2.imencode('.png', multiview_runtime.board_image())
    if not encoded:
        return jsonify({'error': 'Could not draw the ChArUco board.'}), 500
    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype='image/png',
        download_name='charuco-board.png',
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

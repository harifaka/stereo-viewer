import json
import logging
import time
from threading import Lock

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

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


def process_disparity(frame_left, frame_right, params):
    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))

    if frame_left.shape[1] <= params['numDisparities'] + params['blockSize']:
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400

    try:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        block_size = params['blockSize']
        stereo = cv2.StereoSGBM_create(
            minDisparity=params['minDisparity'],
            numDisparities=params['numDisparities'],
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=params['disp12MaxDiff'],
            preFilterCap=params['preFilterCap'],
            uniquenessRatio=params['uniquenessRatio'],
            speckleWindowSize=params['speckleWindowSize'],
            speckleRange=params['speckleRange'],
            mode=STEREO_MODES[params['mode']],
        )
        disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16
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
        encoded, buffer = cv2.imencode('.jpg', color_disparity)
        if not encoded:
            app.logger.error('disparity JPEG encoding failed')
            return jsonify({'error': 'Could not encode the disparity image.'}), 500
    except Exception:
        app.logger.exception('disparity processing failed')
        return jsonify({'error': 'Stereo image processing failed; see container logs.'}), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})


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

    result = process_disparity(frame_left, frame_right, params)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

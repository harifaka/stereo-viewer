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
stereo_params = {'numDisparities': 16, 'blockSize': 15}
params_lock = Lock()


def decode_frame(upload):
    if upload is None:
        return None
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
def disparity():
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
        num_disparities = stereo_params['numDisparities']
        block_size = stereo_params['blockSize']

    if frame_left.shape[1] <= num_disparities + block_size:
        app.logger.warning(
            'disparity rejected width=%s num_disparities=%s block_size=%s',
            frame_left.shape[1], num_disparities, block_size
        )
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400

    try:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        disparity_map = stereo.compute(gray_left, gray_right)
        disparity_gray = cv2.convertScaleAbs(disparity_map, alpha=255 / (num_disparities * 16))
        color_disparity = cv2.applyColorMap(disparity_gray, cv2.COLORMAP_TURBO)
        encoded, buffer = cv2.imencode('.jpg', color_disparity)
        if not encoded:
            app.logger.error('disparity JPEG encoding failed')
            return jsonify({'error': 'Could not encode the disparity image.'}), 500
    except Exception:
        app.logger.exception('disparity processing failed')
        return jsonify({'error': 'Stereo image processing failed; see container logs.'}), 500

    app.logger.info(
        'disparity success left_shape=%s right_shape=%s params=%s elapsed_ms=%.1f',
        frame_left.shape,
        frame_right.shape,
        {'numDisparities': num_disparities, 'blockSize': block_size},
        (time.perf_counter() - started) * 1000
    )
    return Response(buffer.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    data = request.get_json(silent=True) or {}
    try:
        num_disparities = int(data.get('numDisparities', stereo_params['numDisparities']))
        block_size = int(data.get('blockSize', stereo_params['blockSize']))
    except (TypeError, ValueError):
        return jsonify({'error': 'Calibration values must be integers.'}), 400
    if not 16 <= num_disparities <= 256 or num_disparities % 16:
        return jsonify({'error': 'numDisparities must be a multiple of 16 between 16 and 256.'}), 400
    if not 5 <= block_size <= 51 or block_size % 2 == 0:
        return jsonify({'error': 'blockSize must be an odd number between 5 and 51.'}), 400
    with params_lock:
        stereo_params.update(numDisparities=num_disparities, blockSize=block_size)
        params = stereo_params.copy()
    app.logger.info('stereo parameters updated params=%s', params)
    return jsonify({'status': 'success', 'params': params})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

from threading import Lock

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
stereo_params = {'numDisparities': 16, 'blockSize': 15}
params_lock = Lock()


def decode_frame(upload):
    if upload is None:
        return None
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/disparity', methods=['POST'])
def disparity():
    frame_left = decode_frame(request.files.get('left'))
    frame_right = decode_frame(request.files.get('right'))
    if frame_left is None or frame_right is None:
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400

    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))

    with params_lock:
        num_disparities = stereo_params['numDisparities']
        block_size = stereo_params['blockSize']

    if frame_left.shape[1] <= num_disparities + block_size:
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity_map = stereo.compute(gray_left, gray_right)
    disparity_gray = cv2.convertScaleAbs(disparity_map, alpha=255 / (num_disparities * 16))
    color_disparity = cv2.applyColorMap(disparity_gray, cv2.COLORMAP_TURBO)

    encoded, buffer = cv2.imencode('.jpg', color_disparity)
    if not encoded:
        return jsonify({'error': 'Could not encode the disparity image.'}), 500
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
    return jsonify({'status': 'success', 'params': params})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

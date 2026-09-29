import json
import logging
import math
import time
from io import BytesIO
from threading import Lock

from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
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
calibration_state = {'captures': [], 'board': None, 'active': None}
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


def process_disparity(frame_left, frame_right, params, return_map=False, measurement_point=None):
    if frame_left.shape[:2] != frame_right.shape[:2]:
        frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))

    if frame_left.shape[1] <= params['numDisparities'] + params['blockSize']:
        return jsonify({'error': 'The selected camera resolution is too narrow for these settings.'}), 400

    with calibration_lock:
        active_calibration = calibration_state['active']
        if active_calibration and active_calibration['image_size'] == (frame_left.shape[1], frame_left.shape[0]):
            map_left = active_calibration['map_left']
            map_right = active_calibration['map_right']
        else:
            map_left = map_right = None
    if map_left is not None:
        frame_left = cv2.remap(frame_left, map_left[0], map_left[1], cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right, map_right[0], map_right[1], cv2.INTER_LINEAR)

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
        if return_map:
            return disparity_map
        measurement = measurement_error = None
        if measurement_point is not None:
            measurement, measurement_error = estimate_point(
                disparity_map, *measurement_point, active_calibration, params
            )
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

    headers = {'Cache-Control': 'no-store'}
    if measurement_point is not None:
        headers['X-Measurement'] = json.dumps({'measurement': measurement, 'error': measurement_error})
    return Response(buffer.tobytes(), mimetype='image/jpeg', headers=headers)


def calibration_status():
    with calibration_lock:
        active = calibration_state['active']
        result = {
            'captures': len(calibration_state['captures']),
            'requiredCaptures': CALIBRATION_VIEWS_REQUIRED,
            'calibrated': active is not None,
        }
        if active:
            result.update({
                'imageSize': list(active['image_size']),
                'baselineMm': round(active['baseline_mm'], 2),
                'focalLengthPx': round(active['focal_length_px'], 2),
                'squareSizeMm': active['square_size_mm'],
                'columns': active['columns'],
                'rows': active['rows'],
            })
        return result


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
            calibration_state.update({'captures': [], 'board': None, 'active': None})
        return jsonify(calibration_status())

    frame_left = decode_frame(request.files.get('left'))
    frame_right = decode_frame(request.files.get('right'))
    if frame_left is None or frame_right is None:
        return jsonify({'error': 'Upload a valid image from each camera.'}), 400
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
        if calibration_state['board'] != board or calibration_state['captures'] and calibration_state['captures'][0][2] != image_size:
            calibration_state.update({'captures': [], 'board': board, 'active': None})
        calibration_state['board'] = board
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

    measurement_point = None
    if not is_preview and ('measureX' in request.form or 'measureY' in request.form):
        try:
            measurement_point = (int(request.form['measureX']), int(request.form['measureY']))
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'Measurement coordinates must be integers.'}), 400
    result = process_disparity(frame_left, frame_right, params, measurement_point=measurement_point)
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

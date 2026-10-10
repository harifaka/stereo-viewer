import math
from threading import Lock

import cv2
import numpy as np

MAX_PROCESSING_WIDTH = 640
CHARUCO_MIN_CORNERS = 6
CHARUCO_VIEWS_REQUIRED = 4
PROCESSING_MODES = ('preview', 'mvs', 'thermal', 'thermal-stereo', 'slam')
DETECTORS = ('ORB', 'SIFT')

DEFAULT_SETTINGS = {
    'mode': 'preview',
    'referenceSlot': 1,
    'detector': 'ORB',
    'rgbSlot': 1,
    'thermalSlots': [2],
    'overlayOpacity': 0.55,
    'pictureInPicture': True,
    'thermalLeftSlot': 1,
    'thermalRightSlot': 2,
    'claheClip': 3.0,
    'slamReferenceSlot': 1,
}

DEFAULT_BOARD = {
    'squaresX': 5,
    'squaresY': 7,
    'squareSizeMm': 30.0,
    'markerSizeMm': 22.0,
    'referenceSlot': 1,
}


def validate_settings(data, current):
    if not isinstance(data, dict):
        raise ValueError('Processing settings must be a JSON object.')
    mode = data.get('mode', current['mode'])
    if mode not in PROCESSING_MODES:
        raise ValueError('mode must be preview, mvs, thermal, thermal-stereo, or slam.')
    detector = data.get('detector', current['detector'])
    if detector not in DETECTORS:
        raise ValueError('detector must be ORB or SIFT.')
    reference_slot = _slot_value(data, 'referenceSlot', current['referenceSlot'])
    rgb_slot = _slot_value(data, 'rgbSlot', current['rgbSlot'])
    thermal_left = _slot_value(data, 'thermalLeftSlot', current['thermalLeftSlot'])
    thermal_right = _slot_value(data, 'thermalRightSlot', current['thermalRightSlot'])
    slam_reference = _slot_value(data, 'slamReferenceSlot', current['slamReferenceSlot'])
    if thermal_left == thermal_right:
        raise ValueError('Thermal stereo needs two different camera slots.')
    thermal_slots = data.get('thermalSlots', current['thermalSlots'])
    if not isinstance(thermal_slots, list) or any(type(slot) is not int or slot not in range(1, 17) for slot in thermal_slots):
        raise ValueError('thermalSlots must contain camera slots from 1 to 16.')
    if len(set(thermal_slots)) != len(thermal_slots):
        raise ValueError('thermalSlots must not contain duplicates.')
    if rgb_slot in thermal_slots:
        raise ValueError('The RGB camera cannot also be a thermal camera.')
    opacity = data.get('overlayOpacity', current['overlayOpacity'])
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)):
        raise ValueError('overlayOpacity must be a number.')
    opacity = float(opacity)
    if not math.isfinite(opacity) or not 0.15 <= opacity <= 0.85:
        raise ValueError('overlayOpacity must be between 0.15 and 0.85.')
    picture_in_picture = data.get('pictureInPicture', current['pictureInPicture'])
    if not isinstance(picture_in_picture, bool):
        raise ValueError('pictureInPicture must be a boolean.')
    clahe_clip = data.get('claheClip', current['claheClip'])
    if isinstance(clahe_clip, bool) or not isinstance(clahe_clip, (int, float)):
        raise ValueError('claheClip must be a number.')
    clahe_clip = float(clahe_clip)
    if not math.isfinite(clahe_clip) or not 1.0 <= clahe_clip <= 8.0:
        raise ValueError('claheClip must be between 1 and 8.')
    return {
        'mode': mode,
        'referenceSlot': reference_slot,
        'detector': detector,
        'rgbSlot': rgb_slot,
        'thermalSlots': sorted(thermal_slots),
        'overlayOpacity': opacity,
        'pictureInPicture': picture_in_picture,
        'thermalLeftSlot': thermal_left,
        'thermalRightSlot': thermal_right,
        'claheClip': clahe_clip,
        'slamReferenceSlot': slam_reference,
    }


def validate_board(data, current):
    if not isinstance(data, dict):
        raise ValueError('ChArUco settings must be a JSON object.')
    squares_x = _int_setting(data, 'squaresX', current['squaresX'], 3, 8, 'squaresX')
    squares_y = _int_setting(data, 'squaresY', current['squaresY'], 3, 8, 'squaresY')
    if (squares_x - 1) * (squares_y - 1) > 50:
        raise ValueError('This dictionary supports at most 50 markers. Use a smaller board.')
    square_size = _float_setting(data, 'squareSizeMm', current['squareSizeMm'], 5, 200, 'squareSizeMm')
    marker_size = _float_setting(data, 'markerSizeMm', current['markerSizeMm'], 4, 200, 'markerSizeMm')
    if marker_size >= square_size:
        raise ValueError('markerSizeMm must be smaller than squareSizeMm.')
    reference_slot = _slot_value(data, 'referenceSlot', current['referenceSlot'])
    return {
        'squaresX': squares_x,
        'squaresY': squares_y,
        'squareSizeMm': square_size,
        'markerSizeMm': marker_size,
        'referenceSlot': reference_slot,
    }


def _slot_value(data, key, fallback):
    value = data.get(key, fallback)
    if type(value) is not int or value not in range(1, 17):
        raise ValueError(f'{key} must be a camera slot from 1 to 16.')
    return value


def _int_setting(data, key, fallback, low, high, label):
    value = data.get(key, fallback)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer from {low} to {high}.')
    return value


def _float_setting(data, key, fallback, low, high, label):
    value = data.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a number.')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be between {low} and {high}.')
    return value


def limit_width(frame, max_width=MAX_PROCESSING_WIDTH):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scaled_height = max(1, int(round(height * (max_width / width))))
    return cv2.resize(frame, (max_width, scaled_height), interpolation=cv2.INTER_AREA)


def as_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def encode_jpeg(image):
    encoded, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not encoded:
        raise RuntimeError('Could not encode the processing image.')
    return buffer.tobytes()


def message_image(text, width=640, height=360):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (23, 26, 24)
    words = text.split()
    lines = []
    current = ''
    for word in words:
        candidate = word if not current else f'{current} {word}'
        if len(candidate) > 48:
            if current:
                lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    y = max(40, height // 2 - 12 * len(lines))
    for line in lines[:8]:
        cv2.putText(image, line[:80], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 242, 237), 1, cv2.LINE_AA)
        y += 28
    return image


def draw_banner(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (20, 24, 20), -1)
    cv2.putText(image, text[:140], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (199, 243, 107), 1, cv2.LINE_AA)
    return image


def stack_vertical(images):
    width = max(image.shape[1] for image in images)
    resized = []
    for image in images:
        if image.shape[1] != width:
            scale = width / image.shape[1]
            image = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
        resized.append(image)
    return np.vstack(resized)


def camera_matrix(width, height, focal_pixels=None):
    focal = float(width if focal_pixels is None else focal_pixels)
    return np.array([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1],
    ], np.float64)


def scale_camera_matrix(matrix, calibrated_size, frame_shape):
    height, width = frame_shape[:2]
    calibrated_width, calibrated_height = calibrated_size
    scaled = np.array(matrix, np.float64).copy()
    scaled[0, :] *= width / calibrated_width
    scaled[1, :] *= height / calibrated_height
    return scaled


class FrameHub:
    def __init__(self):
        self._lock = Lock()
        self._clients = {}

    def update(self, client_id, frames):
        with self._lock:
            stored = self._clients.setdefault(client_id, {})
            for slot, frame in frames.items():
                stored[slot] = frame.copy()

    def snapshot(self, client_id):
        with self._lock:
            stored = self._clients.get(client_id, {})
            return {slot: frame.copy() for slot, frame in stored.items()}

    def clear(self, client_id):
        with self._lock:
            self._clients.pop(client_id, None)


class SlamState:
    def __init__(self):
        self._lock = Lock()
        self.previous = {}
        self.points = {}
        self.path_length = 0.0
        self.rotation_deg = 0.0
        self.signature = None

    def reset_if_needed(self, signature):
        with self._lock:
            if signature != self.signature:
                self.previous = {}
                self.points = {}
                self.path_length = 0.0
                self.rotation_deg = 0.0
                self.signature = signature

    def track(self, slot, gray, reference):
        with self._lock:
            previous = self.previous.get(slot)
            previous_points = self.points.get(slot)
            tracked = 0
            if (
                previous is None
                or previous.shape != gray.shape
                or previous_points is None
                or len(previous_points) < 20
            ):
                self.points[slot] = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
                self.previous[slot] = gray
                return gray, None, None, 0
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, previous_points, None)
            if next_points is None or status is None:
                self.points[slot] = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
                self.previous[slot] = gray
                return gray, None, None, 0
            valid = status.reshape(-1).astype(bool)
            tracked = int(np.count_nonzero(valid))
            start = previous_points[valid]
            finish = next_points[valid]
            if tracked < 30:
                detected = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
                self.points[slot] = detected if detected is not None else finish.reshape(-1, 1, 2)
            else:
                self.points[slot] = finish.reshape(-1, 1, 2)
            if reference and tracked >= 8 and not np.array_equal(previous, gray):
                motion = np.linalg.norm(finish.reshape(-1, 2) - start.reshape(-1, 2), axis=1)
                self.path_length += float(np.median(motion))
                self._accumulate_rotation(start, finish, gray.shape)
            self.previous[slot] = gray
            return previous, start, finish, tracked

    def _accumulate_rotation(self, start, finish, shape):
        height, width = shape
        matrix = camera_matrix(width, height)
        essential, mask = cv2.findEssentialMat(
            start.reshape(-1, 2),
            finish.reshape(-1, 2),
            matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if essential is None:
            return
        points, rotation, translation, _ = cv2.recoverPose(
            essential,
            start.reshape(-1, 2),
            finish.reshape(-1, 2),
            matrix,
        )
        if points < 8:
            return
        cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
        self.rotation_deg += abs(math.degrees(math.acos(cosine)))
        del translation

    def readout(self):
        with self._lock:
            return self.path_length, self.rotation_deg


def detect_features(frame, detector_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if detector_name == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=800)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    else:
        detector = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors, matcher


def match_features(reference, other, detector_name):
    keypoints_left, descriptors_left, matcher = detect_features(reference, detector_name)
    keypoints_right, descriptors_right, _ = detect_features(other, detector_name)
    if (
        descriptors_left is None
        or descriptors_right is None
        or len(keypoints_left) < 8
        or len(keypoints_right) < 8
    ):
        return None
    try:
        pairs = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    except cv2.error:
        return None
    matches = []
    for neighbors in pairs:
        if len(neighbors) == 2 and neighbors[0].distance < 0.75 * neighbors[1].distance:
            matches.append(neighbors[0])
    if len(matches) < 8:
        return None
    points_left = np.float32([keypoints_left[match.queryIdx].pt for match in matches])
    points_right = np.float32([keypoints_right[match.trainIdx].pt for match in matches])
    fundamental, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC, 3.0, 0.99)
    if fundamental is None or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    if int(np.count_nonzero(inliers)) < 8:
        return None
    return {
        'keypoints_left': keypoints_left,
        'keypoints_right': keypoints_right,
        'matches': matches,
        'points_left': points_left,
        'points_right': points_right,
        'inliers': inliers,
        'fundamental': fundamental,
    }


def triangulated_count(points_left, points_right, inliers, matrix_left, matrix_right):
    left = points_left[inliers]
    right = points_right[inliers]
    if len(left) < 8:
        return 0
    essential, _ = cv2.findEssentialMat(left, right, matrix_left, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None:
        return 0
    _, rotation, translation, _ = cv2.recoverPose(essential, left, right, matrix_left)
    projection_left = matrix_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_right = matrix_right @ np.hstack([rotation, translation])
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left.T, right.T)
    depth = homogeneous[3]
    valid_depth = np.abs(depth) > 1e-8
    points = np.full((len(depth), 3), np.nan)
    points[valid_depth] = (homogeneous[:3, valid_depth] / depth[valid_depth]).T
    return int(np.count_nonzero(np.isfinite(points).all(axis=1)))


def matrix_for_slot(calibration, slot, frame):
    if not calibration or slot not in calibration.get('cameras', {}):
        height, width = frame.shape[:2]
        return camera_matrix(width, height)
    camera = calibration['cameras'][slot]
    return scale_camera_matrix(camera['camera_matrix'], calibration['image_sizes'][slot], frame.shape)


def draw_epilines(image, points, fundamental):
    sample = points[:12].reshape(-1, 1, 2)
    if len(sample) == 0:
        return image
    lines = cv2.computeCorrespondEpilines(sample, 1, fundamental).reshape(-1, 3)
    height, width = image.shape[:2]
    drawn = image.copy()
    for a_coef, b_coef, c_coef in lines:
        if abs(b_coef) < 1e-6:
            continue
        y_start = int(round(-c_coef / b_coef))
        y_end = int(round(-(c_coef + a_coef * (width - 1)) / b_coef))
        cv2.line(drawn, (0, y_start), (width - 1, y_end), (80, 220, 255), 1, cv2.LINE_AA)
    return drawn


def render_mvs(frames, settings, calibration):
    reference_slot = settings['referenceSlot']
    if len(frames) < 2:
        return message_image('Multi-view stereo needs at least two camera frames.'), {
            'mode': 'mvs',
            'ok': False,
            'message': 'Multi-view stereo needs at least two camera frames.',
            'matches': 0,
            'inliers': 0,
            'points': 0,
            'pairs': [],
        }
    if reference_slot not in frames:
        return message_image('The reference camera has no frame.'), {
            'mode': 'mvs',
            'ok': False,
            'message': 'The reference camera has no frame.',
            'matches': 0,
            'inliers': 0,
            'points': 0,
            'pairs': [],
        }
    reference = frames[reference_slot]
    panels = []
    pairs = []
    total_matches = 0
    total_inliers = 0
    total_points = 0
    for slot in sorted(frame_slot for frame_slot in frames if frame_slot != reference_slot):
        other = frames[slot]
        if other.shape[:2] != reference.shape[:2]:
            other = cv2.resize(other, (reference.shape[1], reference.shape[0]))
        matched = match_features(reference, other, settings['detector'])
        if matched is None:
            pairs.append({'slot': slot, 'matches': 0, 'inliers': 0, 'points': 0})
            continue
        inliers = matched['inliers']
        match_count = len(matched['matches'])
        inlier_count = int(np.count_nonzero(inliers))
        points = triangulated_count(
            matched['points_left'],
            matched['points_right'],
            inliers,
            matrix_for_slot(calibration, reference_slot, reference),
            matrix_for_slot(calibration, slot, other),
        )
        total_matches += match_count
        total_inliers += inlier_count
        total_points += points
        pairs.append({'slot': slot, 'matches': match_count, 'inliers': inlier_count, 'points': points})
        display = [match for match, keep in zip(matched['matches'], inliers) if keep][:80]
        lined = draw_epilines(other, matched['points_left'][inliers], matched['fundamental'])
        panel = cv2.drawMatches(
            reference,
            matched['keypoints_left'],
            lined,
            matched['keypoints_right'],
            display,
            None,
            matchColor=(199, 243, 107),
            singlePointColor=(80, 80, 80),
            matchesMask=[1] * len(display),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        panels.append(panel)
    if not panels:
        image = message_image('No reliable feature matches. Aim the cameras at a shared, detailed scene.')
        ok = False
        message = 'No reliable feature matches. Aim the cameras at a shared, detailed scene.'
    else:
        image = stack_vertical(panels)
        ok = True
        message = f'{total_inliers} inliers across {len(pairs)} pairs, {total_points} sparse 3D references.'
        draw_banner(image, message)
    return image, {
        'mode': 'mvs',
        'ok': ok,
        'message': message,
        'matches': total_matches,
        'inliers': total_inliers,
        'points': total_points,
        'pairs': pairs,
    }


def render_thermal(frames, settings):
    rgb_slot = settings['rgbSlot']
    thermal_slots = [slot for slot in settings['thermalSlots'] if slot in frames and slot != rgb_slot]
    if rgb_slot not in frames or not thermal_slots:
        return message_image('Thermal fusion needs the RGB camera and at least one thermal camera frame.'), {
            'mode': 'thermal',
            'ok': False,
            'message': 'Thermal fusion needs the RGB camera and at least one thermal camera frame.',
            'aligned': [],
        }
    rgb = frames[rgb_slot]
    composite = rgb.copy()
    aligned = []
    pip_images = []
    for slot in thermal_slots:
        thermal = frames[slot]
        if thermal.shape[:2] != rgb.shape[:2]:
            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
        matched = match_features(thermal, rgb, 'ORB')
        if matched is None:
            aligned.append({'slot': slot, 'ok': False})
            continue
        source = matched['points_left'][matched['inliers']]
        destination = matched['points_right'][matched['inliers']]
        homography, _ = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
        if homography is None:
            aligned.append({'slot': slot, 'ok': False})
            continue
        gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        warped = cv2.warpPerspective(colored, homography, (rgb.shape[1], rgb.shape[0]))
        composite = cv2.addWeighted(composite, 1 - settings['overlayOpacity'], warped, settings['overlayOpacity'], 0)
        aligned.append({'slot': slot, 'ok': True})
        pip_images.append(colored)
    if settings['pictureInPicture'] and pip_images:
        margin = 12
        thumb_width = max(80, rgb.shape[1] // 4)
        y = composite.shape[0] - margin
        for thumb in pip_images:
            scale = thumb_width / thumb.shape[1]
            thumb_height = max(1, int(thumb.shape[0] * scale))
            resized = cv2.resize(thumb, (thumb_width, thumb_height))
            y -= thumb_height
            x = composite.shape[1] - thumb_width - margin
            if y < margin:
                break
            composite[y:y + thumb_height, x:x + thumb_width] = resized
            y -= 8
    aligned_count = sum(1 for item in aligned if item['ok'])
    ok = aligned_count > 0
    message = (
        f'Aligned {aligned_count} of {len(thermal_slots)} thermal views onto the RGB camera.'
        if ok else
        'Could not align the thermal views. Point them at the same detailed scene as the RGB camera.'
    )
    image = composite if ok else message_image(message)
    if ok:
        draw_banner(image, message)
    return image, {'mode': 'thermal', 'ok': ok, 'message': message, 'aligned': aligned}


def render_thermal_stereo(frames, settings, stereo_params, color_map, stereo_mode):
    left_slot = settings['thermalLeftSlot']
    right_slot = settings['thermalRightSlot']
    if left_slot not in frames or right_slot not in frames:
        return message_image('Thermal stereo needs frames from both selected cameras.'), {
            'mode': 'thermal-stereo',
            'ok': False,
            'message': 'Thermal stereo needs frames from both selected cameras.',
        }
    left = frames[left_slot]
    right = frames[right_slot]
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    if left.shape[1] <= stereo_params['numDisparities'] + stereo_params['blockSize']:
        return message_image('The camera frame is too narrow for the current stereo settings.'), {
            'mode': 'thermal-stereo',
            'ok': False,
            'message': 'The camera frame is too narrow for the current stereo settings.',
        }
    clahe = cv2.createCLAHE(clipLimit=settings['claheClip'], tileGridSize=(8, 8))
    gray_left = clahe.apply(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    gray_right = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    block_size = stereo_params['blockSize']
    stereo = cv2.StereoSGBM_create(
        minDisparity=stereo_params['minDisparity'],
        numDisparities=stereo_params['numDisparities'],
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=stereo_params['disp12MaxDiff'],
        preFilterCap=stereo_params['preFilterCap'],
        uniquenessRatio=stereo_params['uniquenessRatio'],
        speckleWindowSize=stereo_params['speckleWindowSize'],
        speckleRange=stereo_params['speckleRange'],
        mode=stereo_mode,
    )
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16
    valid = disparity > stereo_params['minDisparity']
    disparity_gray = np.zeros(disparity.shape, np.uint8)
    disparity_gray[valid] = np.clip(
        (disparity[valid] - stereo_params['minDisparity']) * (255 / stereo_params['numDisparities']),
        0,
        255,
    ).astype(np.uint8)
    colored = cv2.applyColorMap(disparity_gray, color_map)
    colored[~valid] = (0, 0, 0)
    message = 'CLAHE thermal stereo disparity. Metric distance still uses checkerboard calibration on the stereo page.'
    draw_banner(colored, message)
    return colored, {'mode': 'thermal-stereo', 'ok': True, 'message': message}


def render_slam(frames, settings, slam_state):
    if not frames:
        return message_image('Visual SLAM needs at least one camera frame.'), {
            'mode': 'slam',
            'ok': False,
            'message': 'Visual SLAM needs at least one camera frame.',
            'tracked': 0,
            'pathLength': 0,
            'rotationDeg': 0,
        }
    signature = (settings['slamReferenceSlot'], tuple(sorted(frames)))
    slam_state.reset_if_needed(signature)
    panels = []
    tracked_total = 0
    reference_slot = settings['slamReferenceSlot']
    for slot in sorted(frames):
        frame = frames[slot]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, start, finish, tracked = slam_state.track(slot, gray, slot == reference_slot)
        tracked_total += tracked
        panel = frame.copy()
        if start is not None:
            for origin, destination in zip(start.reshape(-1, 2), finish.reshape(-1, 2)):
                start_point = tuple(np.round(origin).astype(int))
                end_point = tuple(np.round(destination).astype(int))
                cv2.arrowedLine(panel, start_point, end_point, (199, 243, 107), 1, cv2.LINE_AA, tipLength=0.3)
        panels.append(panel)
    path_length, rotation = slam_state.readout()
    image = stack_vertical(panels)
    message = f'Tracked {tracked_total} points. Relative path {path_length:.1f}px, rotation {rotation:.1f} deg.'
    draw_banner(image, message)
    return image, {
        'mode': 'slam',
        'ok': True,
        'message': message,
        'tracked': tracked_total,
        'pathLength': round(path_length, 2),
        'rotationDeg': round(rotation, 2),
    }


def make_board(board_settings):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(
        (board_settings['squaresX'], board_settings['squaresY']),
        float(board_settings['squareSizeMm']),
        float(board_settings['markerSizeMm']),
        dictionary,
    )


def detect_charuco(frame, board):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if corners is None or ids is None or len(ids) < CHARUCO_MIN_CORNERS:
        return None
    return {
        'corners': np.asarray(corners, np.float32).reshape(-1, 1, 2),
        'ids': np.asarray(ids, np.int32).reshape(-1, 1),
    }


def object_and_image_points(board, detection):
    chessboard = np.asarray(board.getChessboardCorners(), np.float32).reshape(-1, 3)
    ids = detection['ids'].reshape(-1)
    corners = detection['corners'].reshape(-1, 2)
    pairs = [
        (int(marker_id), corner)
        for marker_id, corner in zip(ids, corners)
        if 0 <= int(marker_id) < len(chessboard)
    ]
    if len(pairs) < CHARUCO_MIN_CORNERS:
        return None
    object_points = np.array([chessboard[marker_id] for marker_id, _ in pairs], np.float32)
    image_points = np.array([corner for _, corner in pairs], np.float32)
    return object_points, image_points


def relative_pose(reference_rvec, reference_tvec, other_rvec, other_tvec):
    reference_rotation, _ = cv2.Rodrigues(np.asarray(reference_rvec, np.float64).reshape(3, 1))
    other_rotation, _ = cv2.Rodrigues(np.asarray(other_rvec, np.float64).reshape(3, 1))
    rotation = other_rotation @ reference_rotation.T
    translation = np.asarray(other_tvec, np.float64).reshape(3, 1) - rotation @ np.asarray(reference_tvec, np.float64).reshape(3, 1)
    return rotation, translation


def calibrate_cameras(board, views, image_sizes, reference_slot):
    samples = {}
    for view in views:
        for slot, detection in view.items():
            points = object_and_image_points(board, detection)
            if points is not None:
                samples.setdefault(slot, []).append(points)
    if reference_slot not in samples or len(samples[reference_slot]) < CHARUCO_VIEWS_REQUIRED:
        raise ValueError('The reference camera needs at least four board views.')
    intrinsics = {}
    for slot, slot_samples in samples.items():
        if len(slot_samples) < CHARUCO_VIEWS_REQUIRED or slot not in image_sizes:
            continue
        try:
            rms, matrix, distortion, _, _ = cv2.calibrateCamera(
                [sample[0] for sample in slot_samples],
                [sample[1] for sample in slot_samples],
                image_sizes[slot],
                None,
                None,
            )
        except cv2.error as error:
            raise ValueError('Could not calibrate a camera from the captured board views.') from error
        intrinsics[slot] = {
            'camera_matrix': matrix,
            'distortion': distortion,
            'rms': float(rms),
        }
    if reference_slot not in intrinsics:
        raise ValueError('Could not calibrate the reference camera.')
    calibrated = {
        reference_slot: {
            **intrinsics[reference_slot],
            'rotation': np.eye(3),
            'translation': np.zeros((3, 1)),
        }
    }
    for slot in intrinsics:
        if slot == reference_slot:
            continue
        shared = []
        for view in views:
            if slot not in view or reference_slot not in view:
                continue
            reference_points = object_and_image_points(board, view[reference_slot])
            other_points = object_and_image_points(board, view[slot])
            if reference_points is None or other_points is None:
                continue
            shared.append((min(len(reference_points[0]), len(other_points[0])), reference_points, other_points))
        if len(shared) < CHARUCO_VIEWS_REQUIRED:
            continue
        _, reference_points, other_points = max(shared, key=lambda item: item[0])
        reference_ok, reference_rvec, reference_tvec = cv2.solvePnP(
            reference_points[0],
            reference_points[1],
            intrinsics[reference_slot]['camera_matrix'],
            intrinsics[reference_slot]['distortion'],
        )
        other_ok, other_rvec, other_tvec = cv2.solvePnP(
            other_points[0],
            other_points[1],
            intrinsics[slot]['camera_matrix'],
            intrinsics[slot]['distortion'],
        )
        if not reference_ok or not other_ok:
            continue
        rotation, translation = relative_pose(reference_rvec, reference_tvec, other_rvec, other_tvec)
        calibrated[slot] = {
            **intrinsics[slot],
            'rotation': rotation,
            'translation': translation,
        }
    if len(calibrated) < 2:
        raise ValueError('At least two cameras must share four board views.')
    return calibrated


class MultiViewRuntime:
    def __init__(self):
        self._settings_lock = Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.hub = FrameHub()
        self.slam = SlamState()
        self._status_lock = Lock()
        self._status = {}
        self._charuco_lock = Lock()
        self.board = dict(DEFAULT_BOARD)
        self.views = []
        self.image_sizes = {}
        self.calibration = None
        self.charuco_message = 'Show the ChArUco board to at least two cameras and capture a view.'

    def update_settings(self, data):
        with self._settings_lock:
            updated = validate_settings(data, self.settings)
            if updated['mode'] != self.settings['mode'] or updated['slamReferenceSlot'] != self.settings['slamReferenceSlot']:
                self.slam.reset_if_needed(None)
            self.settings = updated
            return dict(updated)

    def current_settings(self):
        with self._settings_lock:
            return dict(self.settings)

    def remember_status(self, client_id, status):
        with self._status_lock:
            self._status[client_id] = status

    def status_for(self, client_id):
        with self._status_lock:
            status = self._status.get(client_id)
        if status is None:
            settings = self.current_settings()
            return {
                'mode': settings['mode'],
                'ok': False,
                'message': 'Processing has not produced a frame yet.',
            }
        return status

    def prepare_frames(self, frames):
        prepared = {}
        for slot, frame in frames.items():
            if frame is None:
                continue
            prepared[int(slot)] = limit_width(as_bgr(frame))
        return prepared

    def render(self, frames, stereo_params, color_map, stereo_mode):
        settings = self.current_settings()
        prepared = self.prepare_frames(frames)
        mode = settings['mode']
        try:
            if mode == 'preview':
                image = message_image('Preview shows the live grid without server processing.')
                status = {'mode': mode, 'ok': True, 'message': 'Preview shows the live grid without server processing.'}
            elif mode == 'mvs':
                image, status = render_mvs(prepared, settings, self.calibration_snapshot())
            elif mode == 'thermal':
                image, status = render_thermal(prepared, settings)
            elif mode == 'thermal-stereo':
                image, status = render_thermal_stereo(prepared, settings, stereo_params, color_map, stereo_mode)
            else:
                image, status = render_slam(prepared, settings, self.slam)
        except Exception as error:
            image = message_image('Processing failed. Check that the selected cameras are connected.')
            status = {'mode': mode, 'ok': False, 'message': str(error) or 'Processing failed.'}
        return encode_jpeg(image), status

    def calibration_snapshot(self):
        with self._charuco_lock:
            if self.calibration is None:
                return None
            return {
                'image_sizes': dict(self.image_sizes),
                'cameras': self.calibration,
            }

    def update_board(self, data):
        with self._charuco_lock:
            updated = validate_board(data, self.board)
            if updated != self.board:
                self.views = []
                self.image_sizes = {}
                self.calibration = None
                self.charuco_message = 'Board settings changed. Capture new views.'
            self.board = updated
            return self._charuco_status_locked()

    def capture(self, frames):
        prepared = self.prepare_frames(frames)
        with self._charuco_lock:
            board = make_board(self.board)
            detected = {}
            for slot, frame in prepared.items():
                detection = detect_charuco(frame, board)
                if detection is None:
                    continue
                image_size = (frame.shape[1], frame.shape[0])
                previous = self.image_sizes.get(slot)
                if previous is not None and previous != image_size:
                    raise ValueError('Camera resolution changed. Reset calibration and capture again.')
                detected[slot] = detection
                self.image_sizes[slot] = image_size
            if len(detected) < 2:
                raise ValueError('At least two cameras must see the ChArUco board.')
            self.views.append(detected)
            self.charuco_message = f'Captured view {len(self.views)}. {len(detected)} cameras saw the board.'
            return self._charuco_status_locked()

    def calibrate(self):
        with self._charuco_lock:
            board = make_board(self.board)
            calibrated = calibrate_cameras(board, self.views, self.image_sizes, self.board['referenceSlot'])
            self.calibration = calibrated
            slots = ', '.join(str(slot) for slot in sorted(calibrated))
            self.charuco_message = f'Calibrated cameras {slots} in the reference camera frame.'
            return self._charuco_status_locked()

    def reset_charuco(self):
        with self._charuco_lock:
            self.views = []
            self.image_sizes = {}
            self.calibration = None
            self.charuco_message = 'ChArUco calibration cleared.'
            return self._charuco_status_locked()

    def charuco_status(self):
        with self._charuco_lock:
            return self._charuco_status_locked()

    def board_image(self):
        with self._charuco_lock:
            board = make_board(self.board)
            width = self.board['squaresX'] * 120
            height = self.board['squaresY'] * 120
        return board.generateImage((width, height))

    def _charuco_status_locked(self):
        detections = {}
        for view in self.views:
            for slot in view:
                detections[slot] = detections.get(slot, 0) + 1
        return {
            **self.board,
            'views': len(self.views),
            'viewsRequired': CHARUCO_VIEWS_REQUIRED,
            'detections': detections,
            'calibrated': self.calibration is not None,
            'cameras': sorted(self.calibration) if self.calibration else [],
            'message': self.charuco_message,
        }

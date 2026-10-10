import logging
import math
import os
from collections import deque
from threading import Lock

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ('auto', 'tapir', 'lucas-kanade')
TAPIR_RESOLUTION = 256
TAPIR_WINDOW = 8
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 30, 0.01),
)
FORWARD_BACKWARD_LIMIT_PX = 1.5
TRAIL_LENGTH = 12

_tapir_lock = Lock()
_tapir_model = None
_tapir_error = None


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def inference_device():
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def tapir_checkpoint_path():
    return os.environ.get('TAPNET_CHECKPOINT', 'models/tapir_checkpoint.pt')


def tapir_availability():
    """Returns (available, message) without loading the model."""
    if _tapir_model is not None:
        return True, f'TAPIR loaded on {inference_device()}.'
    if _tapir_error:
        return False, _tapir_error
    try:
        import importlib.util
        if importlib.util.find_spec('tapnet') is None or importlib.util.find_spec('torch') is None:
            return False, 'The tapnet and torch packages are not installed.'
    except (ImportError, ValueError):
        return False, 'The tapnet and torch packages are not installed.'
    if not os.path.isfile(tapir_checkpoint_path()):
        return False, f'TAPIR checkpoint not found at {tapir_checkpoint_path()}.'
    return True, 'TAPIR is installed and loads on first use.'


def _load_tapir():
    global _tapir_model, _tapir_error
    with _tapir_lock:
        if _tapir_model is not None:
            return _tapir_model
        if _tapir_error:
            raise RuntimeError(_tapir_error)
        try:
            import torch
            from tapnet.torch import tapir_model
            device = inference_device()
            model = tapir_model.TAPIR(pyramid_level=1)
            model.load_state_dict(torch.load(tapir_checkpoint_path(), map_location=device))
            model.to(device).eval()
            _tapir_model = model
            logger.info('TAPIR model loaded on %s', device)
            return model
        except Exception as error:
            _tapir_error = f'TAPIR could not be loaded: {error}'[:240]
            logger.exception('TAPIR model load failed')
            raise RuntimeError(_tapir_error) from error


class LucasKanadeBackend:
    """Pyramidal Lucas-Kanade with a forward-backward consistency check."""

    name = 'lucas-kanade'

    def reset(self):
        pass

    def track(self, previous_gray, gray, points, frame_bgr=None):
        if len(points) == 0:
            return points.copy(), np.zeros(0, bool)
        start = points.reshape(-1, 1, 2).astype(np.float32)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, start, None, **LK_PARAMS)
        if forward is None or status is None:
            return points.copy(), np.zeros(len(points), bool)
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, forward, None, **LK_PARAMS)
        if backward is None or back_status is None:
            return forward.reshape(-1, 2), np.zeros(len(points), bool)
        error = np.linalg.norm(backward.reshape(-1, 2) - start.reshape(-1, 2), axis=1)
        moved = forward.reshape(-1, 2)
        height, width = gray.shape[:2]
        inside = (moved[:, 0] >= 0) & (moved[:, 1] >= 0) & (moved[:, 0] < width) & (moved[:, 1] < height)
        visible = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & (error < FORWARD_BACKWARD_LIMIT_PX)
            & inside
        )
        return moved, visible


class TapirBackend:
    """Google DeepMind TAPIR (TAP-Net family) over a sliding window of recent frames.

    Each step queries the points at the previous frame of the window and reads
    their position and visibility in the newest frame, so occlusion reasoning
    uses the whole temporal window rather than a single frame pair.
    """

    name = 'tapir'

    def __init__(self):
        self.model = _load_tapir()
        self.frames = deque(maxlen=TAPIR_WINDOW)
        self.shape = None

    def reset(self):
        self.frames.clear()
        self.shape = None

    def track(self, previous_gray, gray, points, frame_bgr=None):
        import torch
        if frame_bgr is None:
            frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.shape != frame_bgr.shape[:2]:
            self.frames.clear()
            self.shape = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.frames.append(cv2.resize(rgb, (TAPIR_RESOLUTION, TAPIR_RESOLUTION), interpolation=cv2.INTER_AREA))
        if len(points) == 0:
            return points.copy(), np.zeros(0, bool)
        if len(self.frames) < 2:
            return points.copy(), np.ones(len(points), bool)
        height, width = frame_bgr.shape[:2]
        scale_x = TAPIR_RESOLUTION / width
        scale_y = TAPIR_RESOLUTION / height
        video = np.stack(self.frames).astype(np.float32) / 255.0 * 2.0 - 1.0
        query_frame = len(self.frames) - 2
        query = np.stack([
            np.full(len(points), query_frame, np.float32),
            points[:, 1] * scale_y,
            points[:, 0] * scale_x,
        ], axis=1).astype(np.float32)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(video)[None].to(device),
                torch.from_numpy(query)[None].to(device),
            )
            tracks = output['tracks'][0, :, -1].float().cpu().numpy()
            occlusion = output['occlusion'][0, :, -1]
            expected_dist = output['expected_dist'][0, :, -1]
            visible = ((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected_dist)) > 0.5).cpu().numpy()
        tracks[:, 0] /= scale_x
        tracks[:, 1] /= scale_y
        inside = (tracks[:, 0] >= 0) & (tracks[:, 1] >= 0) & (tracks[:, 0] < width) & (tracks[:, 1] < height)
        return tracks.astype(np.float32), visible.astype(bool) & inside


def create_backend(preference='auto'):
    """Returns (backend, message). Falls back to Lucas-Kanade when TAPIR is unavailable."""
    if preference not in BACKENDS:
        raise ValueError('backend must be auto, tapir, or lucas-kanade.')
    if preference == 'lucas-kanade':
        return LucasKanadeBackend(), 'Lucas-Kanade optical flow selected.'
    available, message = tapir_availability()
    if available:
        try:
            return TapirBackend(), f'TAPIR temporal point tracking on {inference_device()}.'
        except RuntimeError as error:
            message = str(error)
    return LucasKanadeBackend(), f'{message} Using Lucas-Kanade fallback.'


class PointTrackSet:
    """Persistent point tracks that survive short occlusions with constant-velocity prediction."""

    def __init__(self, backend, max_points=80, hold_frames=12):
        self.backend = backend
        self.max_points = max_points
        self.hold_frames = hold_frames
        self.reset()

    def reset(self):
        self.previous = None
        self.positions = np.zeros((0, 2), np.float32)
        self.velocities = np.zeros((0, 2), np.float32)
        self.missing = np.zeros(0, np.int32)
        self.ids = np.zeros(0, np.int64)
        self.trails = {}
        self.next_id = 1
        self.backend.reset()

    def configure(self, max_points, hold_frames):
        self.max_points = max_points
        self.hold_frames = hold_frames

    def _seed(self, gray):
        wanted = self.max_points - len(self.positions)
        if wanted < max(4, self.max_points // 5):
            return
        mask = np.full(gray.shape[:2], 255, np.uint8)
        for x_value, y_value in self.positions:
            cv2.circle(mask, (int(x_value), int(y_value)), 10, 0, -1)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=wanted, qualityLevel=0.01, minDistance=10, mask=mask)
        if corners is None:
            return
        corners = corners.reshape(-1, 2).astype(np.float32)
        count = len(corners)
        new_ids = np.arange(self.next_id, self.next_id + count, dtype=np.int64)
        self.next_id += count
        self.positions = np.vstack([self.positions, corners])
        self.velocities = np.vstack([self.velocities, np.zeros((count, 2), np.float32)])
        self.missing = np.concatenate([self.missing, np.zeros(count, np.int32)])
        self.ids = np.concatenate([self.ids, new_ids])
        for track_id, point in zip(new_ids, corners):
            self.trails[int(track_id)] = deque([tuple(point)], maxlen=TRAIL_LENGTH)

    def step(self, frame_bgr):
        gray = to_gray(frame_bgr)
        if self.previous is None or self.previous.shape != gray.shape:
            self.reset()
            self.previous = gray
            if hasattr(self.backend, 'frames'):
                self.backend.track(gray, gray, np.zeros((0, 2), np.float32), frame_bgr)
            self._seed(gray)
            return
        if len(self.positions):
            moved, visible = self.backend.track(self.previous, gray, self.positions, frame_bgr)
            velocity = moved - self.positions
            self.velocities[visible] = velocity[visible]
            self.positions[visible] = moved[visible]
            self.missing[visible] = 0
            hidden = ~visible
            self.positions[hidden] += self.velocities[hidden]
            self.velocities[hidden] *= 0.8
            self.missing[hidden] += 1
            height, width = gray.shape[:2]
            inside = (
                (self.positions[:, 0] >= 0) & (self.positions[:, 1] >= 0)
                & (self.positions[:, 0] < width) & (self.positions[:, 1] < height)
            )
            keep = inside & (self.missing <= self.hold_frames)
            for track_id in self.ids[~keep]:
                self.trails.pop(int(track_id), None)
            self.positions = self.positions[keep]
            self.velocities = self.velocities[keep]
            self.missing = self.missing[keep]
            self.ids = self.ids[keep]
            for track_id, point in zip(self.ids, self.positions):
                self.trails[int(track_id)].append(tuple(point))
        self.previous = gray
        self._seed(gray)

    def summary(self):
        visible = int(np.count_nonzero(self.missing == 0))
        return {'tracks': int(len(self.positions)), 'visible': visible, 'predicted': int(len(self.positions)) - visible}

    def draw(self, image):
        canvas = image.copy()
        for track_id, point, missing in zip(self.ids, self.positions, self.missing):
            trail = self.trails.get(int(track_id))
            color = (107, 243, 199) if missing == 0 else (60, 160, 255)
            if trail and len(trail) > 1:
                polyline = np.round(np.array(trail)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [polyline], False, color, 1, cv2.LINE_AA)
            center = (int(round(point[0])), int(round(point[1])))
            if missing == 0:
                cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, center, 4, color, 1, cv2.LINE_AA)
        return canvas


class DisparityTemporalFilter:
    """Motion-gated temporal smoothing and short-term hole filling for disparity maps."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        self.age = None
        self.previous_gray = None

    def apply(self, disparity, gray, min_disparity, smoothing, hold_frames, motion_threshold=14):
        if self.state is None or self.state.shape != disparity.shape:
            self.state = disparity.copy()
            self.age = np.where(disparity > min_disparity, 0, hold_frames + 1).astype(np.int32)
            self.previous_gray = gray.copy()
            return disparity, {'smoothedPixels': 0, 'filledPixels': 0}
        valid = np.isfinite(disparity) & (disparity > min_disparity)
        motion = cv2.absdiff(gray, self.previous_gray) > motion_threshold
        motion = cv2.dilate(motion.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        remembered = (self.age <= hold_frames) & (self.state > min_disparity)
        tolerance = np.maximum(1.5, 0.15 * np.abs(self.state))
        consistent = np.abs(disparity - self.state) < tolerance
        blend = valid & remembered & ~motion & consistent
        fill = ~valid & remembered & ~motion
        output = disparity.copy()
        output[blend] = smoothing * self.state[blend] + (1 - smoothing) * disparity[blend]
        output[fill] = self.state[fill]
        self.age = np.where(valid, 0, np.where(fill, self.age + 1, hold_frames + 1)).astype(np.int32)
        self.state = output.copy()
        self.previous_gray = gray.copy()
        return output, {
            'smoothedPixels': int(np.count_nonzero(blend)),
            'filledPixels': int(np.count_nonzero(fill)),
        }


class MeasurementTracker:
    """Follows the user-selected measurement point and holds its last distance through occlusions."""

    def __init__(self):
        self.backend = LucasKanadeBackend()
        self.reset()

    def reset(self):
        self.point = None
        self.velocity = np.zeros(2, np.float32)
        self.previous = None
        self.missing = 0
        self.last_measurement = None
        self.measurement_age = 0

    def use_backend(self, backend):
        if backend.name != self.backend.name:
            self.backend = backend
            self.reset()

    def update(self, requested, frame_bgr, hold_frames):
        gray = to_gray(frame_bgr)
        requested = np.array(requested, np.float32)
        reseed = (
            self.point is None
            or self.previous is None
            or self.previous.shape != gray.shape
            or np.linalg.norm(np.round(self.point) - requested) > 1.5
        )
        if reseed:
            self.reset()
            self.point = requested.copy()
            self.previous = gray
            self.backend.reset()
            if hasattr(self.backend, 'frames'):
                self.backend.track(gray, gray, np.zeros((0, 2), np.float32), frame_bgr)
            return {'x': int(requested[0]), 'y': int(requested[1]), 'tracked': False, 'occluded': False}
        moved, visible = self.backend.track(self.previous, gray, self.point.reshape(1, 2), frame_bgr)
        if visible[0]:
            self.velocity = moved[0] - self.point
            self.point = moved[0].astype(np.float32)
            self.missing = 0
        else:
            self.point = self.point + self.velocity
            self.velocity *= 0.8
            self.missing += 1
        height, width = gray.shape[:2]
        self.point = np.clip(self.point, [0, 0], [width - 1, height - 1]).astype(np.float32)
        self.previous = gray
        if self.missing > hold_frames:
            self.missing = hold_frames
        return {
            'x': int(round(float(self.point[0]))),
            'y': int(round(float(self.point[1]))),
            'tracked': True,
            'occluded': self.missing > 0,
        }

    def stabilize(self, measurement, error, smoothing, hold_frames):
        """Smooths the reported distance and holds the last good value when the match drops out."""
        if measurement is not None:
            previous = self.last_measurement
            if previous is not None and abs(measurement['disparity'] - previous['disparity']) < max(2.0, 0.2 * previous['disparity']):
                measurement = dict(measurement)
                measurement['disparity'] = round(smoothing * previous['disparity'] + (1 - smoothing) * measurement['disparity'], 2)
                measurement['distanceM'] = round(smoothing * previous['distanceM'] + (1 - smoothing) * measurement['distanceM'], 3)
            self.last_measurement = measurement
            self.measurement_age = 0
            return measurement, None
        if self.last_measurement is not None and self.measurement_age < hold_frames:
            self.measurement_age += 1
            held = dict(self.last_measurement)
            held['held'] = True
            return held, None
        return None, error


class StereoDriftCorrector:
    """Markerless online correction of vertical stereo misalignment.

    Rectified pairs should share image rows. Small rig bumps or thermal expansion
    show up as a vertical offset, scale, or slight rotation between the rows of
    matched features. The corrector fits that residual as an affine row model
    and progressively warps the right image so the rows line up again.
    """

    def __init__(self, interval=8, gain=0.5):
        self.interval = interval
        self.gain = gain
        self.reset()

    def reset(self):
        self.matrix = np.eye(3, dtype=np.float64)
        self.shape = None
        self.frames = 0
        self.updates = 0
        self.vertical_error = None
        self.matches = 0

    def _estimate(self, gray_left, gray_right):
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints_left, descriptors_left = orb.detectAndCompute(gray_left, None)
        keypoints_right, descriptors_right = orb.detectAndCompute(gray_right, None)
        if descriptors_left is None or descriptors_right is None:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors_left, descriptors_right)
        if len(matches) < 30:
            return None
        left = np.float32([keypoints_left[match.queryIdx].pt for match in matches])
        right = np.float32([keypoints_right[match.trainIdx].pt for match in matches])
        height = gray_left.shape[0]
        disparity = left[:, 0] - right[:, 0]
        vertical = left[:, 1] - right[:, 1]
        candidates = (disparity >= -2) & (np.abs(vertical) < 0.08 * height)
        if np.count_nonzero(candidates) < 30:
            return None
        right = right[candidates]
        vertical = vertical[candidates]
        design = np.column_stack([right[:, 0], right[:, 1], np.ones(len(right))])
        inliers = np.ones(len(right), bool)
        coefficients = np.zeros(3)
        for _ in range(3):
            if np.count_nonzero(inliers) < 20:
                return None
            coefficients, *_ = np.linalg.lstsq(design[inliers], vertical[inliers], rcond=None)
            residual = np.abs(design @ coefficients - vertical)
            limit = max(1.0, 2.5 * float(np.median(residual[inliers])))
            inliers = residual < limit
        return coefficients, float(np.median(np.abs(vertical[inliers]))), int(np.count_nonzero(inliers))

    def correct(self, gray_left, gray_right, right_frame):
        height, width = gray_right.shape[:2]
        if self.shape != (height, width):
            self.reset()
            self.shape = (height, width)
        corrected_frame = self._warp(right_frame)
        corrected_gray = self._warp(gray_right)
        if self.frames % self.interval == 0:
            estimate = self._estimate(gray_left, corrected_gray)
            if estimate is not None:
                (slope_x, slope_y, offset), error, inliers = estimate
                self.vertical_error = error
                self.matches = inliers
                if error > 0.25:
                    residual = np.array([
                        [1, 0, 0],
                        [slope_x * self.gain, 1 + slope_y * self.gain, offset * self.gain],
                        [0, 0, 1],
                    ], np.float64)
                    updated = residual @ self.matrix
                    if abs(updated[1, 2]) < 0.15 * height and abs(updated[1, 1] - 1) < 0.05 and abs(updated[1, 0]) < 0.05:
                        self.matrix = updated
                        self.updates += 1
                        corrected_frame = self._warp(right_frame)
                        corrected_gray = self._warp(gray_right)
        self.frames += 1
        return corrected_frame, corrected_gray

    def _warp(self, image):
        if np.allclose(self.matrix, np.eye(3)):
            return image
        return cv2.warpAffine(
            image, self.matrix[:2], (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def status(self):
        return {
            'verticalErrorPx': None if self.vertical_error is None else round(self.vertical_error, 2),
            'offsetPx': round(float(self.matrix[1, 2]), 2),
            'rotationDeg': round(math.degrees(math.atan(float(self.matrix[1, 0]))), 3),
            'scale': round(float(self.matrix[1, 1]), 4),
            'updates': self.updates,
            'matches': self.matches,
        }


def mosaic(frames, tile_height=240):
    images = [frames[slot] for slot in sorted(frames)]
    if not images:
        return None
    tiles = []
    for image in images:
        scale = tile_height / image.shape[0]
        tiles.append(cv2.resize(image, (max(1, int(image.shape[1] * scale)), tile_height)))
    columns = int(math.ceil(math.sqrt(len(tiles))))
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = int(math.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * tile_height:(row + 1) * tile_height, column * tile_width:column * tile_width + tile.shape[1]] = tile
    return canvas


def _label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (20, 24, 20), -1)
    cv2.putText(image, text[:120], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (199, 243, 107), 1, cv2.LINE_AA)
    return image


class TapRuntime:
    """Live temporal point tracking for the multi-camera monitor's selected targets."""

    def __init__(self):
        self._lock = Lock()
        self._tracks = {}
        self._backend_preference = None
        self._backend_message = ''
        self._status = {}

    def reset(self):
        with self._lock:
            self._tracks = {}

    def _track_set(self, key, settings):
        if self._backend_preference != settings['backend']:
            self._tracks = {}
            self._backend_preference = settings['backend']
        track_set = self._tracks.get(key)
        if track_set is None:
            backend, self._backend_message = create_backend(settings['backend'])
            track_set = PointTrackSet(backend, settings['maxPoints'], settings['holdFrames'])
            self._tracks[key] = track_set
        else:
            track_set.configure(settings['maxPoints'], settings['holdFrames'])
        return track_set

    def render(self, frames, settings):
        if not settings['enabled']:
            return message_panel('TAP-Net tracking is disabled. Enable it and save the tracking targets.'), {
                'ok': False, 'message': 'Tracking is disabled.', 'targets': [],
            }
        targets = [(f'camera-{slot}', f'Camera {slot}', frames[slot]) for slot in settings['cameraSlots'] if slot in frames]
        if settings['composite'] and frames:
            targets.append(('composite', 'Composite', mosaic(frames)))
        if not targets:
            return message_panel('No frames are available for the selected tracking targets.'), {
                'ok': False, 'message': 'No frames are available for the selected tracking targets.', 'targets': [],
            }
        panels = {}
        summaries = []
        with self._lock:
            wanted = {key for key, _, _ in targets}
            for key in list(self._tracks):
                if key not in wanted:
                    del self._tracks[key]
            for index, (key, label, frame) in enumerate(targets):
                track_set = self._track_set(key, settings)
                try:
                    track_set.step(frame)
                except Exception as error:
                    logger.exception('temporal tracking failed for %s', key)
                    if track_set.backend.name == 'tapir':
                        track_set.backend = LucasKanadeBackend()
                        track_set.reset()
                        self._backend_message = f'TAPIR failed ({error}); using Lucas-Kanade fallback.'[:240]
                    continue
                summary = track_set.summary()
                summaries.append({'target': key, 'label': label, **summary})
                panel = track_set.draw(frame)
                panels[index] = _label(panel, f'{label}: {summary["visible"]} visible, {summary["predicted"]} predicted ({track_set.backend.name})')
            backend_name = next(iter(self._tracks.values())).backend.name if self._tracks else 'lucas-kanade'
            message = self._backend_message
        image = mosaic(panels, tile_height=360) if panels else message_panel('Tracking failed. See the application log.')
        status = {
            'ok': bool(panels),
            'backend': backend_name,
            'message': message,
            'targets': summaries,
        }
        return image, status

    def remember_status(self, client_id, status):
        with self._lock:
            self._status[client_id] = status

    def status_for(self, client_id):
        with self._lock:
            return self._status.get(client_id, {'ok': False, 'message': 'Tracking has not produced a frame yet.', 'targets': []})


def message_panel(text, width=640, height=360):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (23, 26, 24)
    cv2.putText(image, text[:70], (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 242, 237), 1, cv2.LINE_AA)
    if len(text) > 70:
        cv2.putText(image, text[70:140], (20, height // 2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 242, 237), 1, cv2.LINE_AA)
    return image

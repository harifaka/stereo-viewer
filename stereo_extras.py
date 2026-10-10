from threading import Condition

import cv2
import numpy as np

OUTPUT_VIEWS = ('disparity', 'anaglyph', 'bokeh')
MESH_METHODS = ('grid', 'poisson')
MAX_POINT_DISTANCE_M = 15.0


def anaglyph(left, right):
    """Red channel from the left camera, green and blue from the right camera."""
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    composite = right.copy()
    composite[:, :, 2] = left[:, :, 2]
    return composite


def bokeh(frame, disparity, threshold_disparity, blur_size):
    """Keeps pixels at or above the disparity threshold sharp and blurs the background."""
    blur_size = max(3, int(blur_size) | 1)
    foreground = (np.isfinite(disparity) & (disparity >= threshold_disparity)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    alpha = cv2.GaussianBlur(foreground, (21, 21), 0).astype(np.float32)[..., None] / 255.0
    background = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    return (frame.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def reproject(disparity, q_matrix, color_frame, min_disparity, max_distance_m=MAX_POINT_DISTANCE_M):
    """Returns the organized XYZ grid in metres and the mask of valid points."""
    points = cv2.reprojectImageTo3D(disparity.astype(np.float32), q_matrix, handleMissingValues=False) / 1000.0
    distance = np.linalg.norm(points, axis=2)
    mask = (
        np.isfinite(disparity)
        & (disparity > min_disparity)
        & np.isfinite(points).all(axis=2)
        & (points[:, :, 2] > 0)
        & (distance < max_distance_m)
    )
    colors = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    return points.astype(np.float32), colors, mask


def ply_point_cloud(xyz, rgb):
    vertices = np.empty(len(xyz), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'], vertices['y'], vertices['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices['red'], vertices['green'], vertices['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'comment Stereo Lab point cloud, metres, OpenCV camera frame (X right, Y down, Z forward)\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    )
    return header.encode('ascii') + vertices.tobytes()


def ply_mesh(xyz, rgb, faces):
    vertices = np.empty(len(xyz), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'], vertices['y'], vertices['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices['red'], vertices['green'], vertices['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    face_records = np.empty(len(faces), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
    face_records['count'] = 3
    face_records['indices'] = faces
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'comment Stereo Lab mesh, metres, OpenCV camera frame (X right, Y down, Z forward)\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        f'element face {len(face_records)}\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    )
    return header.encode('ascii') + vertices.tobytes() + face_records.tobytes()


def grid_mesh(points, colors, mask, step=2, max_depth_jump=0.05):
    """Triangulates neighbouring valid depth pixels and skips faces across depth discontinuities."""
    points = points[::step, ::step]
    colors = colors[::step, ::step]
    mask = mask[::step, ::step]
    height, width = mask.shape
    index = np.full((height, width), -1, np.int64)
    index[mask] = np.arange(int(np.count_nonzero(mask)))
    top_left = index[:-1, :-1]
    top_right = index[:-1, 1:]
    bottom_left = index[1:, :-1]
    bottom_right = index[1:, 1:]
    depth = points[:, :, 2]

    def continuous(*corners):
        values = np.stack([depth[corner] for corner in corners])
        return (values.max(axis=0) - values.min(axis=0)) <= max_depth_jump * values.min(axis=0)

    rows, columns = np.mgrid[0:height - 1, 0:width - 1]
    corner_tl = (rows, columns)
    corner_tr = (rows, columns + 1)
    corner_bl = (rows + 1, columns)
    corner_br = (rows + 1, columns + 1)
    first = (top_left >= 0) & (bottom_left >= 0) & (top_right >= 0) & continuous(corner_tl, corner_bl, corner_tr)
    second = (top_right >= 0) & (bottom_left >= 0) & (bottom_right >= 0) & continuous(corner_tr, corner_bl, corner_br)
    faces = np.concatenate([
        np.stack([top_left[first], bottom_left[first], top_right[first]], axis=1),
        np.stack([top_right[second], bottom_left[second], bottom_right[second]], axis=1),
    ]).astype(np.int32)
    return points[mask], colors[mask], faces


def poisson_mesh(xyz, rgb, depth=8):
    """Poisson surface reconstruction via Open3D. Raises ImportError when Open3D is not installed."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    cloud = cloud.voxel_down_sample(0.005)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
    vertices = np.asarray(mesh.vertices, np.float32)
    colors = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 200, np.uint8)
    return vertices, colors, np.asarray(mesh.triangles, np.int32)


def open3d_available():
    import importlib.util
    return importlib.util.find_spec('open3d') is not None


class OutputHub:
    """Holds the latest rendered stereo JPEG for any number of MJPEG viewers."""

    def __init__(self):
        self._condition = Condition()
        self._frame = None
        self._sequence = 0

    def publish(self, jpeg):
        with self._condition:
            self._frame = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def frames(self, idle_jpeg):
        seen = -1
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._sequence != seen, timeout=5)
                frame = self._frame
                seen = self._sequence
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + (frame or idle_jpeg) + b'\r\n'

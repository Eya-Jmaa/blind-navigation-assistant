"""3D reasoning: where things are relative to the user, and what blocks the walking path.

Pipeline: depth map -> 3D points -> find the ground plane -> anything sticking up
from the ground inside a corridor in front of the user is an obstacle, whatever it is.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from blindnav.detector import Detection

CAMERA_HEIGHT = 1.4         # m, phone held at chest height
FOV_DEGREES = 70            # field of view along the longer image side (typical phone camera)
PATH_HALF_WIDTH = 0.5       # m -> a 1 m wide corridor in front of the user
PATH_LENGTH = 4.0           # m, how far ahead we look
MIN_OBSTACLE_HEIGHT = 0.25  # m above ground; lower = curb or bump, not a wall
MAX_OBSTACLE_HEIGHT = 2.0   # m, includes head-level obstacles
HEAD_LEVEL = 1.0            # m, obstacle with nothing below this = a white cane can't feel it
MIN_OBSTACLE_POINTS = 15    # ignore isolated noisy points


def focal_length(shape: tuple) -> float:
    """Focal length in pixels, from the image size and the field of view (pinhole camera)."""
    return (max(shape[:2]) / 2) / np.tan(np.radians(FOV_DEGREES) / 2)


def clock_direction(x: float, image_width: int, f: float) -> int:
    """Pixel column -> clock position (12 = straight ahead, 1-2 = right, 10-11 = left)."""
    angle = np.degrees(np.arctan((x - image_width / 2) / f))
    hour = round(angle / 30)  # one clock hour = 30 degrees
    return 12 if hour == 0 else hour % 12


def backproject(depth: np.ndarray, f: float, step: int = 8):
    """Turn a depth map into 3D points (X right, Y down, Z forward), one every `step` pixels."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    Z = depth[ys, xs]
    X = (xs - W / 2) * Z / f  # pinhole model: pixel offset * depth / focal length
    Y = (ys - H / 2) * Z / f
    return np.stack([X, Y, Z], axis=-1), ys, xs


def fit_plane_ransac(points: np.ndarray, iterations: int, rng: np.random.Generator):
    """RANSAC: fit a plane to points, ignoring outliers (people, cars, walls...).

    Repeatedly pick 3 random points, build the plane through them, count how many
    other points lie close to it. The plane with the most supporters wins.
    """
    best, best_count = None, 0
    tolerance = 0.03 * np.median(points[:, 2])  # farther points are noisier, scale with distance
    for _ in range(iterations):
        p1, p2, p3 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue  # the 3 points were aligned, no unique plane
        normal /= norm
        if normal[1] < 0:
            normal = -normal  # make the normal point down (+Y), toward the ground
        if normal[1] < 0.8:
            continue  # not horizontal enough to be the ground (probably a wall)
        offset = -normal @ p1
        count = (np.abs(points @ normal + offset) < tolerance).sum()
        if count > best_count:
            best, best_count = (normal, offset), count
    return best, best_count / len(points)


class GroundEstimator:
    """Finds the ground each frame, and uses it to correct the depth scale.

    Monocular depth gets the *shape* of the scene right but often the *scale* wrong
    (everything 2-3x too far). We know the camera is ~1.4 m above the ground, so if
    the model says 4.2 m, every distance must be divided by 3.
    """

    def __init__(self, iterations: int = 100, history: int = 90):
        self.iterations = iterations
        self.rng = np.random.default_rng(0)
        self.plane = None  # last valid (normal, offset), in raw model units
        self.scale = 1.0
        # Per-frame estimates are noisy (shadows, crowds hiding the ground). The camera height
        # barely changes, so we use the median of the last ~3 s of estimates: very stable,
        # and immune to the occasional bad fit.
        self.recent_scales = deque(maxlen=history)

    def update(self, points: np.ndarray) -> bool:
        lower = points[int(points.shape[0] * 0.6):].reshape(-1, 3)  # ground is at the bottom
        lower = lower[np.isfinite(lower).all(axis=1) & (lower[:, 2] > 0)]
        if len(lower) < 50:
            return self.plane is not None

        plane, inlier_ratio = fit_plane_ransac(lower, self.iterations, self.rng)
        if plane is not None and inlier_ratio > 0.25 and -plane[1] > 0:
            self.plane = plane
            self.recent_scales.append(CAMERA_HEIGHT / -plane[1])  # -offset = camera height
            self.scale = float(np.median(self.recent_scales))
        # If this frame's fit is bad, keep using the last good ground.
        return self.plane is not None


@dataclass
class PathStatus:
    ground_found: bool = False
    blocked: bool = False
    distance: float | None = None   # m to the nearest obstacle in the path
    clock: int | None = None
    head_level: bool = False        # obstacle with nothing below 1 m: invisible to a cane
    approach_speed: float | None = None  # m/s (filled in by risk analysis)
    ttc: float | None = None             # s
    obstacle_pixels: tuple = field(default_factory=lambda: (np.array([]), np.array([])))
    corridor: np.ndarray | None = None  # 4 image points of the path, for drawing


class SpatialAnalyzer:
    def __init__(self):
        self.ground = GroundEstimator()

    def analyze(self, depth: np.ndarray, detections: list[Detection]) -> PathStatus:
        H, W = depth.shape
        f = focal_length(depth.shape)
        points, ys, xs = backproject(depth, f)

        if not self.ground.update(points):
            return PathStatus()
        normal, offset = self.ground.plane
        scale = self.ground.scale

        # 1. Objects found by YOLO: corrected distance, clock position, in the path or not.
        for det in detections:
            if det.distance is None:
                continue
            det.distance *= scale
            cx = (det.box[0] + det.box[2]) / 2
            det.clock = clock_direction(cx, W, f)
            lateral = (cx - W / 2) * det.distance / f  # meters left/right of the user
            det.in_path = abs(lateral) < PATH_HALF_WIDTH + 0.3 and det.distance < PATH_LENGTH

        # 2. Any obstacle, known or not: points sticking up from the ground, inside the corridor.
        P = points * scale
        height = -(points @ normal + offset) * scale  # meters above the ground
        in_corridor = (np.abs(P[..., 0]) < PATH_HALF_WIDTH) & (P[..., 2] > 0) & (P[..., 2] < PATH_LENGTH)
        is_obstacle = in_corridor & (height > MIN_OBSTACLE_HEIGHT) & (height < MAX_OBSTACLE_HEIGHT)

        status = PathStatus(ground_found=True, corridor=self._corridor(normal, offset, scale, f, W, H))
        if is_obstacle.sum() >= MIN_OBSTACLE_POINTS:
            Z = P[..., 2][is_obstacle]
            nearest = float(np.percentile(Z, 5))  # 5th percentile: robust "closest" value
            front = Z < nearest + 0.5             # the front face of the nearest obstacle
            status.blocked = True
            status.distance = nearest
            status.clock = clock_direction(np.median(xs[is_obstacle][front]), W, f)
            status.head_level = bool(height[is_obstacle][front].min() > HEAD_LEVEL)
            status.obstacle_pixels = (ys[is_obstacle], xs[is_obstacle])
        return status

    @staticmethod
    def _corridor(normal, offset, scale, f, W, H) -> np.ndarray:
        """Project the 4 corners of the walking corridor (on the ground) into the image."""
        corners = []
        for x_m, z_m in [(-PATH_HALF_WIDTH, 0.8), (PATH_HALF_WIDTH, 0.8),
                         (PATH_HALF_WIDTH, PATH_LENGTH), (-PATH_HALF_WIDTH, PATH_LENGTH)]:
            X, Z = x_m / scale, z_m / scale  # back to raw model units
            Y = -(normal[0] * X + normal[2] * Z + offset) / normal[1]  # point on the ground plane
            corners.append((f * X / Z + W / 2, f * Y / Z + H / 2))
        return np.array(corners, dtype=np.int32)

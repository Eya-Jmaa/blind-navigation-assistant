"""Motion over time: is each object getting closer, and when would it reach the user?"""

from collections import deque

import numpy as np

from blindnav.detector import Detection
from blindnav.spatial import PathStatus

WINDOW = 1.0              # s of history used to estimate motion
MIN_SAMPLES = 5           # need at least this many measurements for a reliable estimate
MIN_APPROACH_SPEED = 0.3  # m/s, slower than this = not really approaching (noise)
MAX_TTC = 10.0            # s, beyond this = not a threat worth tracking
MIN_BOX_HEIGHT = 40       # px: on smaller boxes, 2 px of jitter already looks like "growth"
FORGET_AFTER = 2.0        # s without seeing an object -> drop its history
PATH_KEY = "path"         # history key for the unknown obstacle in the walking path


class MotionTracker:
    def __init__(self):
        # key -> recent (time, value) measurements
        self.history: dict[int | str, deque] = {}
        self.last_seen: dict[int | str, float] = {}

    def update(self, t: float, detections: list[Detection], path: PathStatus | None,
               image_height: int) -> None:
        """`t` is the frame timestamp in seconds (video time, not wall-clock time)."""
        for det in detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.box
            if y2 - y1 < MIN_BOX_HEIGHT:
                continue  # too small (far away) for a reliable looming estimate
            if y2 >= image_height - 2:
                # Box cut by the bottom of the image: its height no longer grows with the
                # object, so looming would underestimate the danger. Keep the last estimate.
                continue
            # Looming: TTC = size / growth rate = 1 / (d ln(size) / dt), no depth needed.
            growth = self._slope(det.track_id, t, np.log(y2 - y1))
            if growth is not None and growth > 1 / MAX_TTC:
                det.ttc = 1 / growth
                if det.distance is not None:
                    det.approach_speed = det.distance * growth

        # The path obstacle has no box, so we use its depth: TTC = distance / closing speed.
        # As long as the path stays blocked we treat it as one object; when it clears,
        # the history restarts.
        if path is not None and path.blocked:
            slope = self._slope(PATH_KEY, t, path.distance)
            if slope is not None and -slope > MIN_APPROACH_SPEED:
                path.approach_speed = -slope
                path.ttc = path.distance / path.approach_speed
        else:
            self.history.pop(PATH_KEY, None)

        for key in [k for k, seen in self.last_seen.items() if t - seen > FORGET_AFTER]:
            self.history.pop(key, None)
            self.last_seen.pop(key, None)

    def _slope(self, key, t: float, value: float) -> float | None:
        """Rate of change of `value` per second over the last WINDOW seconds."""
        h = self.history.setdefault(key, deque())
        h.append((t, value))
        self.last_seen[key] = t
        while h and t - h[0][0] > WINDOW:
            h.popleft()  # keep only the last WINDOW seconds
        if len(h) < MIN_SAMPLES or h[-1][0] - h[0][0] < WINDOW / 2:
            return None
        # Slope of the line that best fits value vs time (least squares). Much more robust
        # than (v_now - v_before) / dt, which amplifies the noise of just 2 frames.
        times, values = np.array(h).T
        return float(np.polyfit(times, values, deg=1)[0])

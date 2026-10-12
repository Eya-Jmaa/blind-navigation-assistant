"""Drawing overlays on frames for the demo video."""

from collections import defaultdict, deque

import cv2
import numpy as np

from blindnav.detector import Detection
from blindnav.spatial import PathStatus

# One fixed color per class (BGR, because OpenCV uses Blue-Green-Red order).
COLORS = {
    "person": (80, 200, 255),
    "car": (60, 60, 240), "truck": (60, 60, 240), "bus": (60, 60, 240),
    "bicycle": (240, 160, 60), "motorcycle": (240, 160, 60),
    # Street hazards from our own model (phase 6): orange family
    "stairs": (0, 140, 255), "pole": (0, 200, 255), "pillar": (0, 200, 255),
    "cone": (0, 100, 255), "obstacle": (0, 170, 255), "bin": (0, 170, 255),
}
DEFAULT_COLOR = (200, 200, 200)
DANGER_COLOR = (255, 0, 255)  # magenta: approaching with a short time-to-collision
DANGER_TTC = 3.0              # s


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    # Scale line width and text with the image, so 4K and 720p videos look the same.
    scale = frame.shape[0] / 720
    thickness = max(1, round(2 * scale))

    for det in detections:
        x1, y1, x2, y2 = det.box
        color = COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if det.in_path:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness * 2)  # red = in my way
        if det.ttc is not None and det.ttc < DANGER_TTC:
            color = DANGER_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness * 3)

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.distance is not None:
            label = f"{det.class_name} {det.distance:.1f}m"
        if det.clock is not None:
            label += f" @{det.clock}h"
        if det.ttc is not None:
            label += f" TTC {det.ttc:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)  # filled label bg
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 0, 0), thickness)
    return frame


def draw_path(frame: np.ndarray, status: PathStatus) -> np.ndarray:
    scale = frame.shape[0] / 720
    if status.corridor is not None:
        # Semi-transparent corridor: green = clear, red = blocked.
        color = (0, 0, 255) if status.blocked else (0, 200, 0)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [status.corridor], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
        cv2.polylines(frame, [status.corridor], True, color, max(1, round(2 * scale)))

    ys, xs = status.obstacle_pixels
    for y, x in zip(ys, xs):
        cv2.circle(frame, (int(x), int(y)), max(1, round(2 * scale)), (0, 0, 255), -1)

    if not status.ground_found:
        text, color = "PATH: no ground found", (0, 200, 255)
    elif status.blocked:
        text = f"PATH BLOCKED: {status.distance:.1f}m @{status.clock}h"
        if status.ttc is not None:
            text += f" TTC {status.ttc:.1f}s"
        if status.head_level:
            text += " HEAD LEVEL!"
        color = (0, 0, 255)
    else:
        text, color = "PATH: clear", (0, 200, 0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, 2)
    x, y = (frame.shape[1] - tw) // 2, frame.shape[0] - int(25 * scale)
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color,
                max(1, round(2 * scale)))
    return frame


class TrailDrawer:
    """Draws the recent trajectory of each tracked object (where its feet were)."""

    def __init__(self, length: int = 30):
        self.trails = defaultdict(lambda: deque(maxlen=length))

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        thickness = max(1, round(2 * frame.shape[0] / 720))
        for det in detections:
            if det.track_id is None:
                continue
            x1, _, x2, y2 = det.box
            trail = self.trails[det.track_id]
            trail.append(((x1 + x2) // 2, y2))  # bottom-center = where it touches the ground
            if len(trail) > 1:
                color = DANGER_COLOR if det.ttc is not None and det.ttc < DANGER_TTC else (255, 255, 0)
                cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, color, thickness)
        return frame


def draw_caption(frame: np.ndarray, text: str, urgent: bool) -> np.ndarray:
    """Subtitle of what the assistant is saying: LinkedIn videos autoplay without sound."""
    scale = frame.shape[0] / 720
    font_scale, thickness = 1.1 * scale, max(2, round(3 * scale))
    # OpenCV's built-in fonts only support ASCII: replace French accents for the subtitle.
    text = text.translate(str.maketrans("éèêàâùûôîç", "eeeaauuoic"))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = (frame.shape[1] - tw) // 2, int(110 * scale)
    bg = (0, 0, 200) if urgent else (40, 40, 40)
    cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    scale = frame.shape[0] / 720
    cv2.putText(frame, f"{fps:.1f} FPS", (15, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * scale, (0, 255, 0), max(1, round(2 * scale)))
    return frame

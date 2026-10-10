"""Drawing overlays on frames for the demo video."""

import cv2
import numpy as np

from blindnav.detector import Detection
from blindnav.spatial import PathStatus

# One fixed color per class (BGR, because OpenCV uses Blue-Green-Red order).
COLORS = {
    "person": (80, 200, 255),
    "car": (60, 60, 240), "truck": (60, 60, 240), "bus": (60, 60, 240),
    "bicycle": (240, 160, 60), "motorcycle": (240, 160, 60),
}
DEFAULT_COLOR = (200, 200, 200)


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

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.distance is not None:
            label = f"{det.class_name} {det.distance:.1f}m"
        if det.clock is not None:
            label += f" @{det.clock}h"
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


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    scale = frame.shape[0] / 720
    cv2.putText(frame, f"{fps:.1f} FPS", (15, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * scale, (0, 255, 0), max(1, round(2 * scale)))
    return frame

"""Drawing overlays on frames for the demo video."""

import cv2
import numpy as np

from blindnav.detector import Detection

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

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)  # filled label bg
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 0, 0), thickness)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    scale = frame.shape[0] / 720
    cv2.putText(frame, f"{fps:.1f} FPS", (15, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * scale, (0, 255, 0), max(1, round(2 * scale)))
    return frame

"""Object detection: what is in front of the user, and where in the image."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# YOLO is trained on COCO (80 classes). Most of them (cups, laptops, pizza...)
# are irrelevant for someone walking in the street, so we keep only these.
NAVIGATION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "parking meter",
    "bench", "chair", "potted plant", "dog", "cat", "suitcase",
}


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels, top-left / bottom-right corners
    class_name: str
    confidence: float  # 0..1, how sure the model is


class Detector:
    def __init__(self, weights: str = "models/yolo11n.pt", conf: float = 0.35,
                 all_classes: bool = False):
        self.model = YOLO(weights)  # downloads the weights on first use
        self.conf = conf
        # model.names maps class id -> name, e.g. {0: "person", 2: "car", ...}
        self.class_ids = None if all_classes else [
            i for i, name in self.model.names.items() if name in NAVIGATION_CLASSES
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # predict() resizes the frame to 640px, runs the network, then applies
        # confidence filtering + NMS (removes duplicate boxes on the same object).
        result = self.model.predict(frame, conf=self.conf, classes=self.class_ids,
                                    verbose=False)[0]

        detections = []
        for box, cls_id, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                     result.boxes.cls.cpu().numpy(),
                                     result.boxes.conf.cpu().numpy()):
            detections.append(Detection(
                box=tuple(int(v) for v in box),
                class_name=self.model.names[int(cls_id)],
                confidence=float(conf),
            ))
        return detections

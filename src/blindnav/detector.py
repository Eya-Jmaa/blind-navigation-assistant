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
    distance: float | None = None  # meters, filled in by the depth estimator
    clock: int | None = None       # direction, 12 = straight ahead (filled in by spatial analysis)
    in_path: bool = False          # is it in the user's walking path?
    track_id: int | None = None    # same object = same id across frames (filled in by the tracker)
    approach_speed: float | None = None  # m/s, > 0 = getting closer (filled in by risk analysis)
    ttc: float | None = None             # time-to-collision in seconds, None = not approaching


class Detector:
    def __init__(self, weights: str = "models/yolo11n.pt", conf: float = 0.35,
                 all_classes: bool = False):
        self.model = YOLO(weights)  # downloads the weights on first use
        self.conf = conf
        # model.names maps class id -> name, e.g. {0: "person", 2: "car", ...}
        self.class_ids = None if all_classes else [
            i for i, name in self.model.names.items() if name in NAVIGATION_CLASSES
        ]

    def detect(self, frame: np.ndarray, track: bool = True) -> list[Detection]:
        # predict() resizes the frame to 640px, runs the network, then applies
        # confidence filtering + NMS (removes duplicate boxes on the same object).
        # track() does the same, then links each box to the boxes of previous frames
        # (ByteTrack) and gives it an id. persist=True keeps the tracker memory between calls.
        if track:
            result = self.model.track(frame, conf=self.conf, classes=self.class_ids,
                                      tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        else:
            result = self.model.predict(frame, conf=self.conf, classes=self.class_ids,
                                        verbose=False)[0]

        boxes = result.boxes
        # A box that was just detected and not confirmed yet as a track has no id.
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        detections = []
        for box, cls_id, conf, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                                               boxes.conf.cpu().numpy(), ids):
            detections.append(Detection(
                box=tuple(int(v) for v in box),
                class_name=self.model.names[int(cls_id)],
                confidence=float(conf),
                track_id=track_id,
            ))
        return detections

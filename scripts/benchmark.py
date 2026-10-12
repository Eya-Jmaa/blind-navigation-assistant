"""Time every stage of the pipeline on a video, to know where the time goes.

    python scripts/benchmark.py data/videos/busy_street_night.mp4
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from blindnav.announcer import Announcer, rank_alerts
from blindnav.depth import DepthEstimator, object_distance
from blindnav.detector import HAZARD_CLASS_CONF, Detector
from blindnav.risk import MotionTracker
from blindnav.spatial import SpatialAnalyzer
from blindnav.visualize import draw_detections, draw_path

WARMUP = 10  # first frames are slow (CUDA initialization, memory allocation): not measured
# Laptop GPUs boost for ~10 s, then settle at a lower sustained clock (power limit).
# Measuring during the boost would give numbers you never get in real use.
SUSTAINED_WARMUP_S = 15


class Timer:
    """Accumulates the duration of named stages. Waits for the GPU before reading the clock:
    GPU work is asynchronous, so without synchronize() the time would be charged to
    whatever stage happens to wait for the GPU next."""

    def __init__(self):
        self.times = defaultdict(list)
        self.last = None

    def start(self):
        torch.cuda.synchronize()
        self.last = time.perf_counter()

    def lap(self, name):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[name].append(now - self.last)
        self.last = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--depth-size", type=int, default=392)
    parser.add_argument("--depth-every", type=int, default=2)
    args = parser.parse_args()

    detector = Detector("models/yolo11s.pt")
    hazards = None
    if Path("models/hazards.pt").exists():
        hazards = Detector("models/hazards.pt", all_classes=True, class_conf=HAZARD_CLASS_CONF,
                           id_offset=100_000)
    depth_estimator = DepthEstimator(input_size=args.depth_size)
    spatial, motion, announcer = SpatialAnalyzer(), MotionTracker(), Announcer()

    cap = cv2.VideoCapture(args.source)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, first = cap.read()
    start = time.time()
    while time.time() - start < SUSTAINED_WARMUP_S:
        detector.model.predict(first, verbose=False)
        depth_estimator.estimate(first)
    depth = None
    timer = Timer()
    for i in range(args.frames + WARMUP):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

        timer.start()
        detections = detector.detect(frame)
        timer.lap("1. detect (COCO)")
        if hazards:
            detections += hazards.detect(frame)
            timer.lap("2. detect (hazards)")
        if depth is None or i % args.depth_every == 0:
            depth = depth_estimator.estimate(frame)
        timer.lap("3. depth")
        for det in detections:
            det.distance = object_distance(depth, det.box)
        path = spatial.analyze(depth, detections)
        timer.lap("4. spatial")
        motion.update(i / fps_video, detections, path, frame.shape[0])
        announcer.decide(i / fps_video, rank_alerts(detections, path))
        timer.lap("5. risk + announcer")
        draw_path(frame, path)
        draw_detections(frame, detections)
        timer.lap("6. drawing")

        if i == WARMUP - 1:
            timer.times.clear()

    print(f"\n{Path(args.source).name}  {frame.shape[1]}x{frame.shape[0]}  "
          f"{len(timer.times['3. depth'])} frames  ({torch.cuda.get_device_name(0)})  "
          f"depth-size={args.depth_size} depth-every={args.depth_every}")
    total = 0.0
    for name in sorted(timer.times):
        ms = 1000 * np.mean(timer.times[name])
        total += ms
        print(f"  {name:22s} {ms:6.1f} ms")
    print(f"  {'TOTAL':22s} {total:6.1f} ms  ->  {1000 / total:4.1f} FPS")


if __name__ == "__main__":
    main()

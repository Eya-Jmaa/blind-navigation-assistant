"""Run the assistant on a video file or a webcam.

Examples:
    python scripts/run_video.py data/videos/busy_street_night.mp4
    python scripts/run_video.py 0                      # webcam
    python scripts/run_video.py <video> --all-classes  # keep all 80 COCO classes
    python scripts/run_video.py <video> --no-depth     # detection only (Phase 1)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from blindnav.depth import DepthEstimator, colorize, object_distance
from blindnav.detector import Detector
from blindnav.risk import MotionTracker
from blindnav.spatial import SpatialAnalyzer
from blindnav.visualize import TrailDrawer, draw_detections, draw_fps, draw_path

MAX_WIDTH = 1280  # downscale 4K input: faster to process and to encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video path, or webcam index (0)")
    parser.add_argument("--weights", default="models/yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--no-depth", action="store_true", help="skip depth estimation")
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    detector = Detector(args.weights, conf=args.conf, all_classes=args.all_classes)
    depth_estimator = None if args.no_depth else DepthEstimator()
    spatial = SpatialAnalyzer()
    motion = MotionTracker()
    trails = TrailDrawer()
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_index = -1

    writer = None
    if isinstance(source, str):
        out_dir = Path("data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "detect" if args.no_depth else "path"
        out_path = out_dir / f"{Path(source).stem}_{suffix}.mp4"

    fps_smooth = 0.0
    times = {"detect": [], "depth": [], "spatial": []}
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video
        frame_index += 1
        # Speeds must use *video* time. We process slower than real time (~9 FPS on a
        # 30 FPS video), so wall-clock time would make everything look 3x too slow.
        t = frame_index / video_fps if isinstance(source, str) else time.perf_counter()

        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))

        start = time.perf_counter()
        detections = detector.detect(frame)
        t_detect = time.perf_counter()

        depth, path = None, None
        t_depth = t_detect
        if depth_estimator:
            depth = depth_estimator.estimate(frame)
            for det in detections:
                det.distance = object_distance(depth, det.box)
            t_depth = time.perf_counter()
            path = spatial.analyze(depth, detections)
            motion.update(t, detections, path, frame.shape[0])
        end = time.perf_counter()

        times["detect"].append(t_detect - start)
        times["depth"].append(t_depth - t_detect)
        times["spatial"].append(end - t_depth)
        # Exponential moving average: a stable FPS number instead of a flickering one.
        elapsed = end - start
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / elapsed) if fps_smooth else 1 / elapsed

        if path is not None:
            draw_path(frame, path)
        trails.draw(frame, detections)
        draw_detections(frame, detections)
        draw_fps(frame, fps_smooth)
        if depth is not None:
            # Camera view and depth map side by side (landscape) or stacked (portrait video).
            depth_view = draw_detections(colorize(depth), detections)
            frame = np.hstack([frame, depth_view]) if w >= h else np.vstack([frame, depth_view])

        if isinstance(source, str):
            if writer is None:  # created on the first frame, once we know the final size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        if not args.no_show:
            preview = frame
            if preview.shape[1] > 1600 or preview.shape[0] > 900:  # fit the window on screen
                s = min(1600 / preview.shape[1], 900 / preview.shape[0])
                preview = cv2.resize(preview, None, fx=s, fy=s)
            cv2.imshow("blindnav", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {out_path}")
    for name, values in times.items():
        if values and values[0] > 0:
            print(f"{name:>8}: {1000 * np.mean(values[5:] or values):.1f} ms/frame")
    print(f"Overall: {fps_smooth:.1f} FPS")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

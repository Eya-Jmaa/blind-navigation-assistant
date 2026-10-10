"""Run the assistant on a video file or a webcam.

Examples:
    python scripts/run_video.py data/videos/busy_street_night.mp4
    python scripts/run_video.py 0                      # webcam
    python scripts/run_video.py <video> --all-classes  # keep all 80 COCO classes
"""

import argparse
import time
from pathlib import Path

import cv2

from blindnav.detector import Detector
from blindnav.visualize import draw_detections, draw_fps

MAX_WIDTH = 1280  # downscale 4K input: faster to process and to encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video path, or webcam index (0)")
    parser.add_argument("--weights", default="models/yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    detector = Detector(args.weights, conf=args.conf, all_classes=args.all_classes)

    writer = None
    if isinstance(source, str):
        out_dir = Path("data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(source).stem}_detect.mp4"

    fps_smooth = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video

        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))

        start = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = time.perf_counter() - start
        # Exponential moving average: a stable FPS number instead of a flickering one.
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / elapsed) if fps_smooth else 1 / elapsed

        draw_detections(frame, detections)
        draw_fps(frame, fps_smooth)

        if isinstance(source, str):
            if writer is None:  # created on the first frame, once we know the final size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("blindnav", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {out_path}")
    print(f"Average detection speed: {fps_smooth:.1f} FPS")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Run the assistant on a video file or a webcam.

Examples:
    python scripts/run_video.py data/videos/busy_street_night.mp4
    python scripts/run_video.py 0                      # webcam
    python scripts/run_video.py <video> --all-classes  # keep all 80 COCO classes
    python scripts/run_video.py <video> --no-depth     # detection only (Phase 1)
    python scripts/run_video.py <video> --lang fr      # speak French

For a video file, the result (with sound) is saved as data/outputs/<name>_demo.mp4.
The preview window is silent: processing is slower than real time, so the voice is
added to the saved video at the end, perfectly synced.
"""

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from blindnav.announcer import URGENT, Announcer, message, rank_alerts
from blindnav.depth import DepthEstimator, colorize, object_distance
from blindnav.detector import HAZARD_CLASS_CONF, Detector
from blindnav.risk import MotionTracker
from blindnav.spatial import SpatialAnalyzer
from blindnav.speech import LiveSpeaker, SoundtrackRecorder
from blindnav.visualize import TrailDrawer, draw_caption, draw_detections, draw_fps, draw_path

MAX_WIDTH = 1280  # downscale 4K input: faster to process and to encode
CAPTION_DURATION = 2.5  # s a spoken message stays on screen as a subtitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video path, or webcam index (0)")
    parser.add_argument("--weights", default="models/yolo11s.pt")
    parser.add_argument("--hazard-weights", default="models/hazards.pt",
                        help="our street-hazard model (phase 6)")
    parser.add_argument("--no-hazards", action="store_true", help="COCO model only")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--no-depth", action="store_true", help="skip depth estimation")
    parser.add_argument("--depth-size", type=int, default=392,
                        help="depth network input (short side). 518 = full quality, slower")
    parser.add_argument("--depth-every", type=int, default=2,
                        help="compute depth every N frames, reuse it in between")
    parser.add_argument("--no-show", action="store_true", help="don't open a preview window")
    parser.add_argument("--lang", choices=["en", "fr"], default="en", help="voice language")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    detector = Detector(args.weights, conf=args.conf, all_classes=args.all_classes)
    # Second model: stairs, poles, pillars, cones... (classes COCO doesn't know).
    hazard_detector = None
    if not args.no_hazards and Path(args.hazard_weights).exists():
        hazard_detector = Detector(args.hazard_weights, conf=args.conf, all_classes=True,
                                   class_conf=HAZARD_CLASS_CONF, id_offset=100_000)
    elif not args.no_hazards:
        print(f"No hazard model at {args.hazard_weights}: COCO classes only.")
    depth_estimator = None if args.no_depth else DepthEstimator(input_size=args.depth_size)
    depth = None
    spatial = SpatialAnalyzer()
    motion = MotionTracker()
    trails = TrailDrawer()
    announcer = Announcer()
    is_file = isinstance(source, str)
    # Video file: build a synced soundtrack. Webcam: speak live.
    audio = SoundtrackRecorder(args.lang) if is_file else LiveSpeaker(args.lang)
    caption = None  # (text, urgent, time it was said)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_index = -1

    writer = None
    if isinstance(source, str):
        out_dir = Path("data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.no_depth:  # detection only: no alerts, so no sound to add
            out_path = out_dir / f"{Path(source).stem}_detect.mp4"
        else:
            # The silent video is only an intermediate file: the final one gets the soundtrack.
            out_path = Path(tempfile.gettempdir()) / f"{Path(source).stem}_silent.mp4"
            demo_path = out_dir / f"{Path(source).stem}_demo.mp4"
            print(f"Preview is silent. The video WITH SOUND will be saved to {demo_path} "
                  "when processing ends (press q to stop early, not Ctrl+C).")

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
        if hazard_detector:
            detections += hazard_detector.detect(frame)
        t_detect = time.perf_counter()

        path = None  # (depth is NOT reset here: it is reused between frames)
        t_depth = t_detect
        if depth_estimator:
            # The scene barely changes between two frames (~9 cm of walking at 15 FPS):
            # recomputing depth every frame is the most expensive way to learn nothing new.
            if depth is None or frame_index % args.depth_every == 0:
                depth = depth_estimator.estimate(frame)
            for det in detections:
                det.distance = object_distance(depth, det.box)
            t_depth = time.perf_counter()
            path = spatial.analyze(depth, detections)
            motion.update(t, detections, path, frame.shape[0])

            alert = announcer.decide(t, rank_alerts(detections, path))
            if alert is not None:
                text = message(alert, args.lang)
                print(f"[{t:6.2f}s] {'URGENT ' if alert.level == URGENT else 'warning'} {text}")
                if is_file:
                    audio.add(t, alert, text)
                else:
                    audio.say(alert, text)
                caption = (text, alert.level == URGENT, t)
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
        if caption and t - caption[2] < CAPTION_DURATION:
            draw_caption(frame, caption[0], caption[1])
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
        if depth_estimator:
            print("Adding the soundtrack...")
            audio.mux(str(out_path), (frame_index + 1) / video_fps, str(demo_path))
            out_path.unlink()
            print(f"Saved (with sound): {demo_path}")
        else:
            print(f"Saved: {out_path}")
    for name, values in times.items():
        if values and values[0] > 0:
            print(f"{name:>8}: {1000 * np.mean(values[5:] or values):.1f} ms/frame")
    print(f"Overall: {fps_smooth:.1f} FPS")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

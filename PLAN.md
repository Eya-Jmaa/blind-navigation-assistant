# Blind Navigation Assistant — Project Plan

An AI assistant that helps visually impaired people walk safely. It sees through a
camera, estimates **what** is around, **where** it is, **how far** it is, and
**whether it is coming toward you**, then speaks only what matters.

## What makes it different from the tutorial version

| Tutorial version | This project |
|---|---|
| Shouts every detected object | Speaks only about **relevant** threats, ranked by priority |
| No sense of distance | Distance in meters (monocular depth estimation) |
| No notion of motion | Tracks objects, detects **approaching** ones, estimates time-to-collision |
| COCO classes only | Custom model for street hazards (stairs, poles, curbs, crosswalks...) |
| Repeats itself | Cooldowns + memory: never repeats the same warning |
| Text only | Spatial audio (beep on the side of the obstacle) + voice |
| — | On-demand scene description ("what's in front of me?") with a vision-language model |
| No evaluation | Measured: mAP, distance error, FPS, false-alert rate, real user feedback |

## Architecture

```
Camera frame
   │
   ├──► Detector + Tracker (YOLO11 + ByteTrack) ──► boxes, classes, track IDs
   │
   ├──► Depth estimator (Depth Anything V2) ─────► depth map (meters)
   │
   ▼
Spatial reasoning ──► for each object: distance, direction (left/center/right),
   │                  in walking path?, approaching speed, time-to-collision
   ▼
Risk ranking ──► priority score per object
   ▼
Announcer ──► cooldowns, dedup, TTS voice + spatial beeps
   │
   └── (button) Scene describer (VLM) ──► "You are at a crosswalk, light is red"
```

## Phases

Each phase ends with something working that you can show (and post about).

| # | Phase | You learn | Where |
|---|---|---|---|
| 0 | Setup + record test videos | Environment, data collection | Local |
| 1 | Detection baseline (pretrained YOLO11) | Inference, classes, confidence, NMS | Local |
| 2 | Depth estimation → distance per object | Monocular depth, relative vs metric depth | Local |
| 3 | Direction + walking corridor | Image geometry, region logic | Local |
| 4 | Tracking + approach / time-to-collision | Multi-object tracking, temporal smoothing | Local |
| 5 | Smart announcer (priority, cooldown, TTS, spatial audio) | Decision logic, real-time UX | Local |
| 6 | Custom hazard model | Dataset building, annotation, fine-tuning, mAP | **Kaggle** |
| 7 | Real-time optimization | Profiling, frame skipping, FP16, ONNX/TensorRT export | Local |
| 8 | Scene description on demand | Vision-language models | Local / API |
| 9 | Evaluation | Metrics, honest error analysis | Local |
| 10 | Demo video, GitHub, LinkedIn series | Communication | — |

## Hardware

- **Local:** NVIDIA RTX 2050 (4 GB). Used for real-time inference (small models: `yolo11n/s`, `Depth-Anything-V2-Small`).
- **Kaggle:** free GPU (T4 / P100) for training the custom model in Phase 6.

## Project structure

```
CV/
├── PLAN.md               ← this file
├── README.md             ← public-facing (written at the end)
├── requirements.txt
├── data/
│   ├── videos/           ← your recorded walking videos (not committed)
│   └── outputs/          ← annotated output videos
├── models/               ← downloaded / trained weights (not committed)
├── notebooks/            ← Kaggle training notebooks
├── src/blindnav/
│   ├── detector.py       ← Phase 1 & 4
│   ├── depth.py          ← Phase 2
│   ├── spatial.py        ← Phase 3
│   ├── risk.py           ← Phase 4
│   ├── announcer.py      ← Phase 5
│   ├── scene.py          ← Phase 8
│   └── visualize.py      ← drawing overlays for the demo
└── scripts/              ← entry points (run on a video / webcam)
```

## Final LinkedIn post — assets to save while building

One post at the end ("I've been working on..."). Save these along the way:

- [ ] Phase 1: clip of the raw baseline (detects everything, no intelligence)
- [ ] Phase 2: depth map side by side with the camera view
- [ ] Phase 4: clip of an approaching person flagged with time-to-collision
- [ ] Phase 5: clip **with sound** — dumb version vs smart announcer
- [ ] Phase 6: training curves + mAP before/after custom model
- [ ] Phase 9: key numbers (FPS, distance error, false-alert rate)
- [ ] One failure + how it was fixed (people love these)
- [ ] Final demo video (30–60 s, readable without sound, subtitles)
- [ ] Quote from a real visually impaired user, if possible

"""Decision-making: out of everything the system perceives, what deserves to be said, and when.

Perception (phases 1-4) says WHAT is there. This module decides what MATTERS right now.
A blind user can only process one short message every ~2 seconds, so silence is the
default and every word must be worth it.
"""

import math
from dataclasses import dataclass

from blindnav.detector import Detection
from blindnav.spatial import PathStatus

WARNING, URGENT = 1, 2
WARNING_THRESHOLD = 0.3   # urgency needed to say something
URGENT_THRESHOLD = 0.6    # urgency needed to interrupt with "Stop!"
PROXIMITY_RANGE = 4.0     # m: in the path, urgency grows from 0 (at 4 m) to 1 (at 0 m)
TTC_RANGE = 6.0           # s: urgency grows from 0 (impact in 6 s) to 1 (impact now)
MAX_DISTANCE = 20.0       # m: farther detections are ignored (mostly false positives)
MAX_APPROACH_DISTANCE = 12.0  # m: beyond this, distances are unreliable -> no "approaching" alert

# How bad a collision would be. A car hurts more than a person or a bench.
CLASS_WEIGHT = {
    "car": 1.0, "bus": 1.0, "truck": 1.0, "train": 1.0, "motorcycle": 1.0,
    "bicycle": 0.9, "person": 0.7, "dog": 0.6,
    # Street hazards (our phase 6 model). Stairs: a fall is among the worst accidents.
    "stairs": 1.0, "pole": 0.8, "pillar": 0.8, "obstacle": 0.7, "cone": 0.6,
}
DEFAULT_WEIGHT = 0.6      # static objects: bench, chair, potted plant, hydrant...
OBSTACLE_WEIGHT = 0.8     # unknown obstacle found by the 3D corridor
HEAD_LEVEL_WEIGHT = 1.0   # a cane can't detect it: treat it as seriously as a car

MIN_PERSISTENCE = 0.2     # s an alert must exist before being spoken (kills 1-frame false positives)
MIN_GAP = 2.0             # s between two messages, so the user can process each one
REPEAT_AFTER = 5.0        # s before repeating an alert about the same thing...
REPEAT_IF_CLOSER = 1.0    # ...and only if it got at least this much closer


@dataclass
class Alert:
    key: int | str        # what the alert is about: a track id, or "path"
    label: str            # COCO class name, or "obstacle"
    distance: float
    clock: int
    urgency: float        # 0..1, used to rank alerts
    level: int            # WARNING or URGENT
    approaching: bool = False
    head_level: bool = False


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def urgency(distance: float, ttc: float | None, in_path: bool, weight: float) -> float:
    """Combine two independent reasons to worry, and keep the strongest one.

    - proximity: something very close IN MY PATH (even if it doesn't move: a pole)
    - time: something that will reach me soon (even if it is still far: a fast car)
    """
    proximity = clip01(1 - distance / PROXIMITY_RANGE) if in_path else 0.0
    time = clip01(1 - ttc / TTC_RANGE) if ttc is not None and distance < MAX_APPROACH_DISTANCE else 0.0
    if not in_path:
        time *= 0.6  # approaching but beside my path: I'll probably walk past it
    return weight * max(proximity, time)


def rank_alerts(detections: list[Detection], path: PathStatus | None) -> list[Alert]:
    """Turn raw perception into a list of alerts, most urgent first."""
    alerts = []
    for det in detections:
        if det.distance is None or det.distance > MAX_DISTANCE or det.track_id is None:
            continue  # no track id = not confirmed over several frames yet
        weight = CLASS_WEIGHT.get(det.class_name, DEFAULT_WEIGHT)
        u = urgency(det.distance, det.ttc, det.in_path, weight)
        alerts.append(Alert(det.track_id, det.class_name, det.distance, det.clock or 12, u,
                            level=0, approaching=det.ttc is not None))

    if path is not None and path.blocked:
        # If YOLO already knows what blocks the path, its name is more useful than "obstacle".
        named = any(d.in_path and d.distance is not None and abs(d.distance - path.distance) < 1.0
                    for d in detections)
        if not named or path.head_level:
            weight = HEAD_LEVEL_WEIGHT if path.head_level else OBSTACLE_WEIGHT
            u = urgency(path.distance, path.ttc, True, weight)
            alerts.append(Alert("path", "obstacle", path.distance, path.clock or 12, u, level=0,
                                approaching=path.ttc is not None, head_level=path.head_level))

    for a in alerts:
        a.level = URGENT if a.urgency >= URGENT_THRESHOLD else (
            WARNING if a.urgency >= WARNING_THRESHOLD else 0)
    return sorted([a for a in alerts if a.level], key=lambda a: a.urgency, reverse=True)


class Announcer:
    """Decides WHEN to speak. Remembers what was said to avoid repeating itself."""

    def __init__(self):
        self.first_seen: dict = {}  # key -> time the alert first appeared (persistence check)
        self.spoken: dict = {}      # key -> (time, level, distance) of the last message about it
        self.last_message_time = -math.inf

    def decide(self, t: float, alerts: list[Alert]) -> Alert | None:
        """Return the one alert to speak now, or None (silence)."""
        present = {a.key for a in alerts}
        self.first_seen = {k: v for k, v in self.first_seen.items() if k in present}
        for a in alerts:
            self.first_seen.setdefault(a.key, t)

        for a in alerts:  # already sorted: most urgent first
            if self._should_say(t, a):
                self.spoken[a.key] = (t, a.level, a.distance)
                self.last_message_time = t
                return a
        return None

    def _should_say(self, t: float, a: Alert) -> bool:
        if t - self.first_seen[a.key] < MIN_PERSISTENCE:
            return False  # might be a 1-frame false positive (remember the hydrant)
        if a.level < URGENT and t - self.last_message_time < MIN_GAP:
            return False  # only an URGENT alert may cut in right after another message

        previous = self.spoken.get(a.key)
        if previous is None:
            return True   # never talked about this one
        when, level, distance = previous
        if a.level > level:
            return True   # escalation: it became more dangerous since we last said it
        # Same danger still there: remind only after a while, and only if it got closer.
        return t - when > REPEAT_AFTER and distance - a.distance > REPEAT_IF_CLOSER


# ----- Message wording -----------------------------------------------------------------

LABELS = {
    "fr": {"person": "personne", "bicycle": "vélo", "car": "voiture", "motorcycle": "moto",
           "bus": "bus", "truck": "camion", "train": "train", "traffic light": "feu",
           "stop sign": "panneau stop", "fire hydrant": "borne incendie",
           "parking meter": "horodateur", "bench": "banc", "chair": "chaise",
           "potted plant": "plante", "dog": "chien", "cat": "chat", "suitcase": "valise",
           "obstacle": "obstacle", "stairs": "escalier", "pole": "poteau", "pillar": "pilier",
           "cone": "cône", "bin": "poubelle"},
}
TEMPLATES = {
    "en": {"ahead": "ahead", "clock": "at {h} o'clock", "near": "very close",
           "meters": "{n} meter", "meters_plural": "{n} meters", "stop": "Stop! ",
           "approaching": "{label} approaching", "head": "Watch your head! "},
    "fr": {"ahead": "devant", "clock": "à {h} heures", "near": "tout près",
           "meters": "{n} mètre", "meters_plural": "{n} mètres", "stop": "Stop ! ",
           "approaching": "{label} qui approche", "head": "Attention à la tête ! "},
}


def message(alert: Alert, lang: str = "en") -> str:
    """Short, always the same structure: [Stop!] <what> <where>, <how far>."""
    T = TEMPLATES[lang]
    label = LABELS.get(lang, {}).get(alert.label, alert.label)
    what = T["approaching"].format(label=label) if alert.approaching else label
    where = T["ahead"] if alert.clock == 12 else T["clock"].format(h=alert.clock)
    n = round(alert.distance)
    far = T["near"] if alert.distance < 1 else T["meters" if n == 1 else "meters_plural"].format(n=n)

    prefix = T["head"] if alert.head_level else (T["stop"] if alert.level == URGENT else "")
    text = f"{prefix}{what} {where}, {far}"
    return text[0].upper() + text[1:]

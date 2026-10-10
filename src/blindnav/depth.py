"""Monocular depth estimation: how far away is each pixel, from a single camera."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# "Metric" = outputs real distances in meters (not just "closer / farther").
# "Outdoor" = fine-tuned on street scenes. "Small" = 25M params, fits real-time on a 4 GB GPU.
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthEstimator:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FP16 (half precision) halves memory and is ~2x faster on GPU, with almost no accuracy loss.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, dtype=self.dtype).to(self.device).eval()

    @torch.inference_mode()  # no gradients needed: we only predict, never train here
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return a (H, W) array: distance in meters for every pixel of the frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # the model was trained on RGB images
        inputs = self.processor(images=rgb, return_tensors="pt")  # resize to ~518px + normalize
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)

        depth = self.model(pixel_values=pixel_values).predicted_depth  # (1, h, w), low resolution
        # Upsample back to the frame size so depth[y, x] matches frame[y, x].
        depth = F.interpolate(depth[None], size=frame.shape[:2], mode="bilinear",
                              align_corners=False)[0, 0]
        return depth.float().cpu().numpy()


def object_distance(depth: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Distance to an object = median depth of the center of its bounding box.

    We shrink the box first: its edges often contain background (wall, road behind
    the person), which would make the object look farther than it is. The median
    (not the mean) ignores the few remaining outlier pixels.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = x1 + w // 4, x2 - w // 4
    cy1, cy2 = y1 + h // 4, y2 - h // 4
    region = depth[max(cy1, 0):cy2, max(cx1, 0):cx2]
    return float(np.median(region)) if region.size else float("nan")


def colorize(depth: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
    """Turn a depth map into a color image: near = bright/yellow, far = dark/purple."""
    norm = 1 - np.clip(depth / max_dist, 0, 1)  # invert so close objects are bright
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

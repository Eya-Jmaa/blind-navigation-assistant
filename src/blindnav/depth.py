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
    def __init__(self, model_id: str = DEFAULT_MODEL, input_size: int = 518):
        """`input_size`: short side of the image fed to the network (518 = training size).
        Smaller is faster (cost grows with the number of pixels) but less detailed."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FP16 (half precision) halves memory and is ~2x faster on GPU, with almost no accuracy loss.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.input_size = input_size
        processor = AutoImageProcessor.from_pretrained(model_id)
        # Normalization constants the model was trained with (ImageNet mean / std).
        self.mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, dtype=self.dtype).to(self.device).eval()

    def _network_size(self, h: int, w: int) -> tuple[int, int]:
        # Keep the aspect ratio, short side = input_size, both sides multiples of 14
        # (the network cuts the image into 14x14 patches).
        scale = self.input_size / min(h, w)
        return (max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14))

    @torch.inference_mode()  # no gradients needed: we only predict, never train here
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return a (H, W) array: distance in meters for every pixel of the frame."""
        # Preprocessing on the GPU (the Hugging Face processor does it on the CPU: 22 ms).
        # Same steps: BGR->RGB, resize (bicubic), scale to 0..1, normalize with mean/std.
        x = torch.from_numpy(frame).to(self.device)            # (H, W, 3) uint8, BGR
        x = x.flip(-1).permute(2, 0, 1)[None].float() / 255    # (1, 3, H, W) RGB in 0..1
        x = F.interpolate(x, size=self._network_size(*frame.shape[:2]), mode="bicubic",
                          align_corners=False, antialias=True)  # antialias = like PIL resize
        x = ((x - self.mean) / self.std).to(self.dtype)

        depth = self.model(pixel_values=x).predicted_depth  # (1, h, w), low resolution
        # Upsample back to the frame size so depth[y, x] matches frame[y, x].
        depth = F.interpolate(depth[None], size=frame.shape[:2], mode="bilinear",
                              align_corners=False)[0, 0]
        # Copy back in float16 (half the bytes of float32), convert on the CPU side.
        return depth.half().cpu().numpy().astype(np.float32)


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

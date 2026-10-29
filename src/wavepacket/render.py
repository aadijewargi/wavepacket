"""Turn complex wavefunctions into images and animations.

The default "phase" style uses domain colouring: hue encodes the complex
phase :math:`\\arg\\psi` and brightness encodes the amplitude, so you can see
both the probability density and the local momentum (the rate at which the
colours cycle) in one picture.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb
from PIL import Image

WALL_RGB = np.array([0.80, 0.83, 0.90])  # outline of walls / barriers
FILL_RGB = np.array([0.13, 0.15, 0.20])  # inside of walls / barriers


def _brightness(amp: np.ndarray, ref: float, gamma: float) -> np.ndarray:
    return np.clip((amp / ref) ** gamma, 0.0, 1.0)


def domain_color(psi: np.ndarray, ref: Optional[float] = None, gamma: float = 0.75,
                 saturation: float = 0.9) -> np.ndarray:
    """Map ``psi`` to an RGB float image (hue = phase, value = |psi|^gamma)."""
    amp = np.abs(psi)
    ref = float(np.percentile(amp, 99.9)) if ref is None else ref
    hue = (np.angle(psi) / (2 * np.pi)) % 1.0
    hsv = np.stack([hue, np.full_like(hue, saturation), _brightness(amp, ref or 1.0, gamma)], -1)
    return hsv_to_rgb(hsv)


def density_color(psi: np.ndarray, ref: Optional[float] = None, gamma: float = 0.5,
                  cmap: str = "inferno") -> np.ndarray:
    """Map ``|psi|^2`` through a sequential colormap."""
    rho = np.abs(psi) ** 2
    ref = float(np.percentile(rho, 99.9)) if ref is None else ref
    return colormaps[cmap](_brightness(rho, ref or 1.0, gamma))[..., :3]


def overlay_potential(rgb: np.ndarray, V: Optional[np.ndarray], vmax: Optional[float] = None,
                      alpha: float = 0.9, fill: Sequence[float] = FILL_RGB,
                      edge: Sequence[float] = WALL_RGB) -> np.ndarray:
    """Shade high-potential regions and outline the contour ``V = vmax/2``."""
    if V is None or not np.any(V > 0):
        return rgb
    vmax = float(V.max()) if vmax is None else vmax
    a = alpha * np.clip(V / vmax, 0.0, 1.0)[..., None]
    out = rgb * (1 - a) + np.asarray(fill) * a
    mask = V >= 0.5 * vmax
    border = np.zeros_like(mask)
    for axis in range(mask.ndim):
        for shift in (1, -1):
            border |= mask & ~np.roll(mask, shift, axis=axis)
    out[border] = 0.25 * out[border] + 0.75 * np.asarray(edge)
    return out


def frame(psi: np.ndarray, V: Optional[np.ndarray] = None, style: str = "phase",
          ref: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Render one 2D frame as a ``uint8`` RGB array, flipped so +y is up."""
    if style == "phase":
        rgb = domain_color(psi, ref)
    elif style == "density":
        rgb = density_color(psi, None if ref is None else ref ** 2)
    else:
        raise ValueError(f"unknown style {style!r} (use 'phase' or 'density')")
    rgb = overlay_potential(rgb, V, vmax)
    return (np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8)


def to_image(rgb: np.ndarray, size: Optional[int] = None) -> Image.Image:
    img = Image.fromarray(rgb)
    if size is not None and size != img.width:
        h = round(img.height * size / img.width)
        img = img.resize((size, h), Image.LANCZOS)
    return img


def save_gif(frames: Iterable[np.ndarray], path: str, fps: int = 25,
             size: Optional[int] = None) -> None:
    """Write RGB uint8 frames to an optimised looping GIF."""
    imgs = [to_image(f, size) for f in frames]
    # One shared adaptive palette keeps colours stable across frames.
    sample = Image.new("RGB", (imgs[0].width, imgs[0].height * min(len(imgs), 6)))
    step = max(1, len(imgs) // 6)
    for i, im in enumerate(imgs[::step][:6]):
        sample.paste(im, (0, i * im.height))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps),
              loop=0, optimize=True, disposal=1)


class SmoothedReference:
    """Exponentially smoothed brightness reference, to avoid GIF flicker."""

    def __init__(self, rate: float = 0.15, percentile: float = 99.8) -> None:
        self.rate, self.percentile, self.value = rate, percentile, None

    def __call__(self, psi: np.ndarray) -> float:
        target = float(np.percentile(np.abs(psi), self.percentile))
        self.value = target if self.value is None else (
            (1 - self.rate) * self.value + self.rate * target)
        return self.value

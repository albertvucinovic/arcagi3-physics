"""Animated GIF export for authoritative ARC Physics Timelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arc_agi.rendering import COLOR_MAP, hex_to_rgb
from PIL import Image, ImageDraw

from .review import Review, frame

_PALETTE = tuple(hex_to_rgb(COLOR_MAP[index]) for index in range(16))


def export_gif(
    review: Review,
    destination: str | Path,
    *,
    scale: int = 8,
    duration_ms: int = 200,
    level_pause_ms: int = 800,
) -> Path:
    """Export every authoritative Timeline observation as a looping GIF."""

    if scale < 1:
        raise ValueError("GIF scale must be positive")
    if duration_ms < 1 or level_pause_ms < 1:
        raise ValueError("GIF frame durations must be positive")
    images = []
    durations = []
    previous_levels = None
    for index in range(review.transitions + 1):
        item = frame(review, index)
        state = item["state"]
        image = _state_image(state, scale, index=index, action=item["action"])
        levels = state.get("levels_completed")
        duration = (
            level_pause_ms
            if previous_levels is not None and levels != previous_levels
            else duration_ms
        )
        images.append(image)
        durations.append(duration)
        previous_levels = levels
    if not images:
        raise ValueError("ARC review has no frames")

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return path


def _state_image(
    state: dict[str, Any], scale: int, *, index: int, action: Any
) -> Image.Image:
    layers = state.get("grid")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("ARC state has no grid layers")
    grid = layers[-1]
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("ARC visible grid is empty")
    width = len(grid[0])
    if width < 1 or any(not isinstance(row, (list, tuple)) or len(row) != width for row in grid):
        raise ValueError("ARC visible grid must be rectangular")
    image = Image.new("RGB", (width, len(grid)))
    pixels = []
    for row in grid:
        for cell in row:
            if type(cell) is not int or not 0 <= cell < len(_PALETTE):
                raise ValueError("ARC grid cells must be integer color indices 0 through 15")
            pixels.append(_PALETTE[cell])
    image.putdata(pixels)
    if scale != 1:
        image = image.resize((width * scale, len(grid) * scale), Image.Resampling.NEAREST)
    label_height = 14
    canvas = Image.new("RGB", (image.width, image.height + label_height), _PALETTE[0])
    canvas.paste(image, (0, label_height))
    action_id = action.get("action") if isinstance(action, dict) else "-"
    label = f"action {index}  input {action_id}  level {state.get('levels_completed', '-')}"
    ImageDraw.Draw(canvas).text((2, 2), label, fill=_PALETTE[5])
    canvas.putpixel((canvas.width - 1, label_height - 1), _marker(index))
    return canvas


def _marker(index: int) -> tuple[int, int, int]:
    value = index % (1 << 24)
    return value >> 16, (value >> 8) & 255, value & 255

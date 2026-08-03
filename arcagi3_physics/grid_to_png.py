from __future__ import annotations

"""Source for ARC's sandbox-local gridToPng.py visualization helper."""

GRID_TO_PNG = r'''#!/usr/bin/env python3
"""Render an ARC-AGI-3 color-index grid as a nearest-neighbor PNG.

Examples:
  python gridToPng.py current-grid.json current-grid.png
  python gridToPng.py canonical-input.json current-grid.png
  cat grid.json | python gridToPng.py - current-grid.png

Accepted JSON is either a 2-D grid, ARC's list of grid layers (the visible last
layer is used), a public state containing ``grid``, or canonical-input.json (the
latest public state is selected from its Timeline).
"""

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

PALETTE = (
    (255, 255, 255),
    (204, 204, 204),
    (153, 153, 153),
    (102, 102, 102),
    (51, 51, 51),
    (0, 0, 0),
    (229, 58, 163),
    (255, 123, 204),
    (249, 60, 49),
    (30, 147, 255),
    (136, 216, 241),
    (255, 220, 0),
    (255, 133, 27),
    (146, 18, 49),
    (79, 204, 48),
    (163, 86, 214),
)


def _is_grid(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and bool(row) for row in value)
        and all(not isinstance(cell, (list, dict)) for row in value for cell in row)
    )


def _public_state(value):
    if not isinstance(value, dict):
        return value
    if "timeline" in value:
        timeline = value["timeline"]
        if not isinstance(timeline, list) or not timeline:
            raise ValueError("timeline must be a non-empty list")
        value = timeline[-1]
    if isinstance(value, dict) and "next_state" in value:
        value = value["next_state"]
    elif (
        isinstance(value, dict)
        and "grid" not in value
        and isinstance(value.get("state"), dict)
    ):
        value = value["state"]
    if isinstance(value, dict) and "grid" in value:
        value = value["grid"]
    return value


def _grid(value):
    value = _public_state(value)
    if _is_grid(value):
        grid = value
    elif isinstance(value, list) and value and all(_is_grid(layer) for layer in value):
        grid = value[-1]
    else:
        raise ValueError("input does not contain a rectangular 2-D ARC grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal length")
    converted = []
    for row in grid:
        converted_row = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < 16:
                raise ValueError("grid cells must be integer ARC color indices 0 through 15")
            converted_row.append(cell)
        converted.append(converted_row)
    return converted


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png(grid, scale):
    height, width = len(grid), len(grid[0])
    rows = []
    for row in grid:
        pixels = b"".join(bytes(PALETTE[cell]) * scale for cell in row)
        scanline = b"\x00" + pixels
        rows.extend([scanline] * scale)
    header = struct.pack(">IIBBBBB", width * scale, height * scale, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + _chunk(b"IEND", b"")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON file, canonical-input.json, or - for stdin")
    parser.add_argument("output", nargs="?", default="grid.png", help="output PNG (default: grid.png)")
    parser.add_argument("--scale", type=int, default=8, help="integer nearest-neighbor scale (default: 8)")
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale must be positive")
    source = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    grid = _grid(json.loads(source))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_png(grid, args.scale))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

ARC_DOMAIN_FILES = (("gridToPng.py", GRID_TO_PNG),)

__all__ = ["ARC_DOMAIN_FILES", "GRID_TO_PNG"]

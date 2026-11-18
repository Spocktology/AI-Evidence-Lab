"""The labelled grid the user marks, and native-resolution crops of what they marked.

Cells are named like a spreadsheet: columns A, B, C... left to right, rows
1, 2, 3... top to bottom. "C4" is the third column, fourth row.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .media import fit_to_budget, fits_budget, max_native_tile


def col_name(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(name: str) -> int:
    n = 0
    for ch in name:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_cell(cell: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Z]+)(\d+)", cell.strip().upper())
    if not m:
        raise ValueError(f"Bad cell name: {cell}")
    return col_index(m.group(1)), int(m.group(2)) - 1


@dataclass
class Grid:
    cols: int
    rows: int
    width: int
    height: int

    @classmethod
    def for_size(cls, width: int, height: int, cols: int) -> "Grid":
        cols = max(2, min(int(cols), 26))
        rows = max(2, int(cols * height / width + 0.5))  # same rounding as the browser (Math.round)
        return cls(cols, rows, width, height)

    def cell_box(self, c: int, r: int) -> tuple[int, int, int, int]:
        x0 = round(c * self.width / self.cols)
        x1 = round((c + 1) * self.width / self.cols)
        y0 = round(r * self.height / self.rows)
        y1 = round((r + 1) * self.height / self.rows)
        return x0, y0, x1, y1

    def all_cells(self) -> list[str]:
        return [f"{col_name(c)}{r + 1}" for r in range(self.rows) for c in range(self.cols)]

    def union_box(self, cells: list[str], pad_cells: float = 0.0) -> tuple[int, int, int, int]:
        boxes = [self.cell_box(*parse_cell(c)) for c in cells]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        px = int(pad_cells * self.width / self.cols)
        py = int(pad_cells * self.height / self.rows)
        return max(0, x0 - px), max(0, y0 - py), min(self.width, x1 + px), min(self.height, y1 + py)

    def as_dict(self) -> dict:
        return {"cols": self.cols, "rows": self.rows, "width": self.width, "height": self.height}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_grid(im: Image.Image, grid: Grid, highlight: list[str] | None = None) -> Image.Image:
    """Overlay the labelled grid (scaled to `im`) and outline highlighted cells."""
    im = im.convert("RGB").copy()
    sx, sy = im.width / grid.width, im.height / grid.height
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fsize = max(10, int(min(im.width / grid.cols, im.height / grid.rows) * 0.18))
    font = _font(fsize)
    hl = set(highlight or [])
    for r in range(grid.rows):
        for c in range(grid.cols):
            x0, y0, x1, y1 = grid.cell_box(c, r)
            box = (x0 * sx, y0 * sy, x1 * sx - 1, y1 * sy - 1)
            name = f"{col_name(c)}{r + 1}"
            if name in hl:
                d.rectangle(box, outline=(255, 60, 60, 255), width=max(2, fsize // 5))
            else:
                d.rectangle(box, outline=(255, 255, 255, 90), width=1)
            d.text((box[0] + 3, box[1] + 2), name, font=font, fill=(255, 255, 0, 200),
                   stroke_width=max(1, fsize // 8), stroke_fill=(0, 0, 0, 200))
    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def native_tiles(im: Image.Image, box: tuple[int, int, int, int], max_tiles: int = 4) -> list[dict]:
    """Cut `box` from the full-resolution image into tiles Claude can read at
    1:1 pixels. If the region would need more than `max_tiles` tiles it is
    downscaled as little as possible and that scale is reported."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if fits_budget(w, h):
        return [{"image": im.crop(box), "box": list(box), "scale": 1.0}]
    side = max_native_tile()
    nx, ny = math.ceil(w / side), math.ceil(h / side)
    if nx * ny <= max_tiles:
        tw, th = math.ceil(w / nx), math.ceil(h / ny)
        tiles = []
        for j in range(ny):
            for i in range(nx):
                b = (x0 + i * tw, y0 + j * th, min(x1, x0 + (i + 1) * tw), min(y1, y0 + (j + 1) * th))
                tiles.append({"image": im.crop(b), "box": list(b), "scale": 1.0})
        return tiles
    crop, scale = fit_to_budget(im.crop(box))
    return [{"image": crop, "box": list(box), "scale": round(scale, 4)}]


def enhance(crop: Image.Image) -> Image.Image:
    """Contrast-limited local equalization + mild gamma lift. Reveals detail in
    dark/low-contrast regions without inventing pixels (no sharpening, no AI
    upscaling). Clearly labelled as enhanced wherever it is shown."""
    import cv2

    arr = np.asarray(crop.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_chan)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if l_chan.mean() < 70:  # dark scene: lift shadows
        lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
        out = lut[out]
    return Image.fromarray(out)


def pixel_zoom(crop: Image.Image, target: int = 1200) -> tuple[Image.Image, int]:
    """Nearest-neighbour enlargement of a small region so individual pixels are
    visible. Nearest-neighbour adds no new information (each pixel becomes a
    block), unlike smooth or AI upscaling."""
    factor = max(1, min(8, target // max(crop.width, crop.height)))
    if factor <= 1:
        return crop, 1
    return crop.resize((crop.width * factor, crop.height * factor), Image.NEAREST), factor


def autocontrast(im: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(im.convert("RGB"), cutoff=0.5)

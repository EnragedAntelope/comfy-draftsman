"""Preview-image downscaling for outputs returned to the MCP client.

A full-size render is easily 1-2 MB of PNG; inlined as base64 image content
that dominates the token cost of a run. Tools that show images therefore
return a thumbnail by default and point at view_output(max_dim=None) for the
full-resolution file.
"""

from __future__ import annotations

import cmath
import io
import math
import statistics
from array import array

from PIL import Image as PILImage

# Formats the MCP image content types cover; anything else is re-encoded.
_PASSTHROUGH_FORMATS = {"png", "jpeg", "webp", "gif"}
_ALREADY_LOSSY = {"jpeg", "webp", "gif"}
JPEG_QUALITY = 85
# An opaque PNG render at even 640px is ~500KB; JPEG q85 is ~10x smaller, so
# big opaque PNGs re-encode even when no resize is needed.
REENCODE_THRESHOLD = 256 * 1024


def downscale_image(data: bytes, max_dim: int | None) -> tuple[bytes, str, int, int]:
    """Return (bytes, format) fit for MCP image content, thumbnailed to max_dim.

    max_dim None/0 means full RESOLUTION (an oversized opaque PNG still
    re-encodes as JPEG - same pixels, ~10x fewer tokens). Downscaled images
    re-encode as JPEG (opaque) or PNG (alpha). Raises ValueError if the payload
    isn't a decodable image (e.g. a video file listed under an image output).
    """
    try:
        img: PILImage.Image = PILImage.open(io.BytesIO(data))
        src_format = (img.format or "png").lower()
        needs_resize = max_dim and max(img.size) > max_dim
    except Exception as e:
        raise ValueError(f"not a decodable image: {e}") from e
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if (
        not needs_resize
        and src_format in _PASSTHROUGH_FORMATS
        and (src_format in _ALREADY_LOSSY or has_alpha or len(data) <= REENCODE_THRESHOLD)
    ):
        return data, src_format, img.width, img.height
    if needs_resize:
        assert max_dim is not None  # needs_resize is only true when max_dim is set
        img.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    if has_alpha:
        img.save(buf, format="PNG")
        return buf.getvalue(), "png", img.width, img.height
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), "jpeg", img.width, img.height


# --- sweep contact sheets ------------------------------------------------------

LABEL_H = 18
CROP_MAX = 512  # per side; a crop tile is only useful at 1:1, so it is capped, never scaled
INLINE_MAX = 1568  # a larger image gets downscaled by the client anyway


def contact_sheet(
    rows: list[list[tuple[PILImage.Image | None, str]]], thumb: int | None
) -> PILImage.Image:
    """Grid of labelled cells, one row per list. ``thumb`` fits every image into a
    square of that size; ``None`` keeps them at 1:1. Cells are sized to the largest
    image AFTER fitting, so portrait/landscape renders don't sit in square cells."""
    from PIL import ImageDraw

    tiles: list[list[tuple[PILImage.Image | None, str]]] = []
    for row in rows:
        out_row = []
        for img, label in row:
            tile = img.convert("RGB") if img is not None else None  # always a copy
            if tile is not None and thumb:
                tile.thumbnail((thumb, thumb), PILImage.Resampling.LANCZOS)
            out_row.append((tile, label))
        tiles.append(out_row)
    images = [t for row in tiles for t, _ in row if t is not None]
    cell_w = max((i.width for i in images), default=96)
    cell_h = max((i.height for i in images), default=96) + LABEL_H
    ncols = max(len(r) for r in tiles)
    sheet = PILImage.new("RGB", (ncols * cell_w, len(tiles) * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, (tile, label) in enumerate(row):
            x, y = c * cell_w, r * cell_h
            if tile is not None:
                sheet.paste(tile, (x, y))
            draw.text((x + 3, y + cell_h - LABEL_H + 3), label[: cell_w // 6], fill=(230, 230, 230))
    return sheet


def _pixel_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel box for an int (pixel) or float (fraction of the image) box. A
    fractional box keeps its centre but is capped at CROP_MAX per side, so the
    same box tracks the subject across resolutions and still shows 1:1."""
    if not any(isinstance(n, float) for n in box):
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    edges = []
    for lo, hi, size in ((box[0], box[2], width), (box[1], box[3], height)):
        a, b = round(lo * size), round(hi * size)
        if b - a > CROP_MAX:
            a = (a + b - CROP_MAX) // 2
            b = a + CROP_MAX
        edges.append((a, b))
    return edges[0][0], edges[1][0], edges[0][1], edges[1][1]


def crop_tiles(img: PILImage.Image, boxes: list[list[float]]) -> list[PILImage.Image | None]:
    """Cut each [x0, y0, x1, y1] out of the full-resolution image, unscaled.
    A box that falls outside the image gives None rather than a bad tile."""
    tiles: list[PILImage.Image | None] = []
    for raw in boxes:
        x0, y0, x1, y1 = _pixel_box(raw, img.width, img.height)
        box = (max(x0, 0), max(y0, 0), min(x1, img.width), min(y1, img.height))
        tiles.append(img.crop(box) if box[2] > box[0] and box[3] > box[1] else None)
    return tiles


# Stripe periods searched, in pixels. 8 px and under is every VAE's normal texture
# (a peak there in clean renders too); MOSS-Image SFT draws ~48 stripes per width.
_STRIPE_PERIODS = (10, 64)
BANDING_VISIBLE = 8.0  # clean real renders score 1-6; visibly striped MOSS SFT 11-16


def _fft(x: list[complex]) -> list[complex]:
    n = len(x)
    if n == 1:
        return x
    even, odd = _fft(x[0::2]), _fft(x[1::2])
    tw = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [e + t for e, t in zip(even, tw, strict=True)] + [e - t for e, t in zip(even, tw, strict=True)]


def _stripe_peak(profiles: list[list[float]]) -> tuple[float, float]:
    """(peak / median of its spectral neighbours, period px) for the strongest
    repeating stripe in these same-length profiles, their spectra summed."""
    n = len(profiles[0]) - 8
    size = 1 << (n - 1).bit_length()
    total = [0.0] * (size // 2)
    for p in profiles:
        hp = [p[i] - sum(p[i - 4 : i + 5]) / 9 for i in range(4, len(p) - 4)]  # drop the ramp
        mean = sum(hp) / n
        spec = _fft([complex(v - mean) for v in hp] + [0j] * (size - n))
        total = [t + abs(c) for t, c in zip(total, spec[: size // 2], strict=True)]
    pad = size / n  # zero padding spreads one real bin over this many
    best = (0.0, 0.0)
    lo, hi = _STRIPE_PERIODS
    for k in range(max(1, math.ceil(size / hi)), min(size // 2, math.floor(size / lo) + 1)):
        # a stripe is a sharp line; image content is a broad bump, so compare with
        # the bins 3-12 real bins away rather than the whole spectrum
        ring = [total[j] for d in range(round(3 * pad), round(13 * pad)) for j in (k - d, k + d) if 0 < j < size // 2]
        if ring and (base := statistics.median(ring)) > 1e-6:
            best = max(best, (total[k] / base, size / k))
    return best


def banding_score(img: PILImage.Image) -> tuple[float, str]:
    """(score, stripe) for the strongest evenly spaced stripe running the full
    height or width, 10-64 px apart, in color (red-green + blue-yellow) or
    brightness. The image is averaged down to one column profile and one row
    profile - linear, so opponent colors come from the RGB profiles - and each
    profile's spectrum peak is compared with its neighbours: about 1-6 for clean
    renders, BANDING_VISIBLE and up for visible stripes (MOSS-Image SFT's
    vertical color bands score 11-16)."""
    rgb = img.convert("RGB")
    best = (0.0, 0.0, "")
    for size, axis in (((rgb.width, 1), "vertical"), ((1, rgb.height), "horizontal")):
        r, g, b = (list(array("f", ch.convert("F").resize(size, PILImage.Resampling.BOX).tobytes())) for ch in rgb.split())
        if len(r) < 2 * _STRIPE_PERIODS[1]:
            continue
        chroma = [[x - y for x, y in zip(r, g, strict=True)], [z - (x + y) / 2 for x, y, z in zip(r, g, b, strict=True)]]
        luma = [[0.299 * x + 0.587 * y + 0.114 * z for x, y, z in zip(r, g, b, strict=True)]]
        for profiles in (chroma, luma):
            score, period = _stripe_peak(profiles)
            best = max(best, (score, period, axis))
    score, period, axis = best
    return round(score, 1), (f"{axis} {period:.0f}px" if axis else "")


def crop_guide(img: PILImage.Image, size: int = 384) -> PILImage.Image:
    """Thumbnail with a labelled 0.1 grid, for aiming fractional sweep crops."""
    from PIL import ImageDraw

    guide = img.convert("RGB")
    guide.thumbnail((size, size), PILImage.Resampling.LANCZOS)
    draw = ImageDraw.Draw(guide)
    for i in range(1, 10):
        x, y = round(guide.width * i / 10), round(guide.height * i / 10)
        color = (255, 220, 0) if i == 5 else (0, 230, 255)
        draw.line([(x, 0), (x, guide.height)], fill=color)
        draw.line([(0, y), (guide.width, y)], fill=color)
        draw.text((x + 2, 2), f".{i}", fill=color)
        draw.text((2, y + 2), f".{i}", fill=color)
    return guide


def to_png(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

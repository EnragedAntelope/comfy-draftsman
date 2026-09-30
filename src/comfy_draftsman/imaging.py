"""Preview-image downscaling for outputs returned to the MCP client.

A full-size render is easily 1-2 MB of PNG; inlined as base64 image content
that dominates the token cost of a run. Tools that show images therefore
return a thumbnail by default and point at view_output(max_dim=None) for the
full-resolution file.
"""

from __future__ import annotations

import io

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
    square of that size; ``None`` keeps them at 1:1 (cells grow to the largest)."""
    from PIL import ImageDraw

    images = [img for row in rows for img, _ in row if img is not None]
    cell_w = thumb or max((i.width for i in images), default=96)
    cell_h = (thumb or max((i.height for i in images), default=96)) + LABEL_H
    ncols = max(len(r) for r in rows)
    sheet = PILImage.new("RGB", (ncols * cell_w, len(rows) * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (img, label) in enumerate(row):
            x, y = c * cell_w, r * cell_h
            if img is not None:
                tile = img.convert("RGB")
                if thumb:
                    tile.thumbnail((thumb, thumb), PILImage.Resampling.LANCZOS)
                sheet.paste(tile, (x, y))
            draw.text((x + 3, y + cell_h - LABEL_H + 3), label[: cell_w // 6], fill=(230, 230, 230))
    return sheet


def crop_tiles(img: PILImage.Image, boxes: list[list[int]]) -> list[PILImage.Image | None]:
    """Cut each [x0, y0, x1, y1] out of the full-resolution image, unscaled.
    A box that falls outside the image gives None rather than a bad tile."""
    tiles: list[PILImage.Image | None] = []
    for x0, y0, x1, y1 in boxes:
        box = (max(x0, 0), max(y0, 0), min(x1, img.width), min(y1, img.height))
        tiles.append(img.crop(box) if box[2] > box[0] and box[3] > box[1] else None)
    return tiles


def to_png(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

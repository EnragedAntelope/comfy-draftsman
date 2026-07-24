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

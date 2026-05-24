# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
from PIL import Image

from librewxr.config import settings
from librewxr.tiles.coordinates import tile_pixel_latlons


def render_satellite_tile(
    cloud_grid,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a satellite-like cloud cover tile from IFS cloud data.

    Composites three cloud layers (high, mid, low) into a semi-transparent
    RGBA tile suitable for overlaying on a base map.  Higher clouds render
    brighter (white) and more opaque, simulating an infrared satellite view
    where cold, high cloud tops appear brightest.

    Args:
        cloud_grid: CloudGrid instance with loaded data.
        z, x, y: Tile coordinates.
        tile_size: 256 or 512.
        timestamp: Unix timestamp to select nearest IFS timestep.
        fmt: "png" or "webp".

    Returns:
        Encoded image bytes.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    high, mid, low = cloud_grid.sample(lat_grid, lon_grid, timestamp)

    # Convert 0-100% to 0.0-1.0 float
    h = high.astype(np.float32) / 100.0
    m = mid.astype(np.float32) / 100.0
    lo = low.astype(np.float32) / 100.0

    # Per-layer opacity weights: high clouds most prominent (IR-like)
    alpha_h = h * 0.80
    alpha_m = m * 0.55
    alpha_l = lo * 0.40

    # Total opacity via over-operator approximation
    total_alpha = 1.0 - (1.0 - alpha_h) * (1.0 - alpha_m) * (1.0 - alpha_l)

    # Weighted brightness: high=white, mid=light gray, low=darker gray
    #   High:  245
    #   Mid:   215
    #   Low:   185
    weight_sum = alpha_h + alpha_m + alpha_l
    has_cloud = weight_sum > 0.001
    brightness = np.where(
        has_cloud,
        (alpha_h * 245.0 + alpha_m * 215.0 + alpha_l * 185.0)
        / np.where(has_cloud, weight_sum, 1.0),
        0.0,
    )

    # Build RGBA
    rgba = np.zeros((*high.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(brightness, 0, 255).astype(np.uint8)
    # Slight cool tint in blue channel
    rgba[..., 2] = np.clip(brightness + 5.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(total_alpha * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


# GMGSI LW encoded value at the cloud threshold — pixels colder than
# this (encoded > threshold) are treated as cloud and rendered opaque;
# warmer pixels (ground / ocean / low cloud) ramp to fully transparent.
# Roughly 270 K on GMGSI's 0–255 brightness-temperature scale.
_LW_CLOUD_THRESHOLD = 110.0
_LW_CLOUD_MAX = 255.0


def _lw_brightness_and_alpha(
    encoded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute float32 brightness + alpha planes for one LW grid.

    Shared between the LW-only renderer and the composite renderer so
    the night side of the composite matches stand-alone LW exactly.
    Returns ``(brightness, alpha)`` both shaped like ``encoded`` with
    values in [0, 255] and [0, 1] respectively.
    """
    no_data = encoded == 0
    cloud_ramp = np.clip(
        (encoded.astype(np.float32) - _LW_CLOUD_THRESHOLD)
        / (_LW_CLOUD_MAX - _LW_CLOUD_THRESHOLD),
        0.0, 1.0,
    )
    alpha = np.power(cloud_ramp, 0.7)
    alpha = np.where(no_data, 0.0, alpha)
    brightness = encoded.astype(np.float32)
    return brightness, alpha


def _pack_rgba(brightness: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build an RGBA uint8 array with a slight cool tint in the blue."""
    rgba = np.zeros((*brightness.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(brightness + 5.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def render_gmgsi_tile(
    source,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a single-channel LW tile.

    Used as the fallback when VIS is unavailable.  Cold cloud tops are
    rendered as opaque bright pixels; warm ground / ocean / low cloud
    fades to transparent via the shared LW threshold ramp.  Same math
    as the LW half of the composite, served stand-alone.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    encoded = source.sample(lat_grid, lon_grid, timestamp)
    brightness, alpha = _lw_brightness_and_alpha(encoded)
    rgba = _pack_rgba(brightness, alpha)
    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def render_gmgsi_composite_tile(
    lw_source,
    vis_source,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a VIS-over-LW composite tile.

    The LW channel forms the base: cold cloud tops on a transparent
    map, same threshold ramp as the stand-alone LW renderer.  The VIS
    channel paints reflected sunlight on top with ``alpha = vis/255``,
    so the day side shows the natural view-from-space (clouds, land,
    ocean) and the night side falls through to LW IR.  The terminator
    crossfade emerges from the underlying VIS reflectance field — no
    sun-angle math required.

    Standard "VIS over LW" alpha composite::

        out_color = vis_color * vis_alpha + lw_color * (1 - vis_alpha)
        out_alpha = vis_alpha + lw_alpha * (1 - vis_alpha)

    Where VIS is zero (night side, or outside the disk), the formula
    collapses to LW alone, which is what we want.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    lw_encoded = lw_source.sample(lat_grid, lon_grid, timestamp)
    vis_encoded = vis_source.sample(lat_grid, lon_grid, timestamp)

    lw_brightness, lw_alpha = _lw_brightness_and_alpha(lw_encoded)

    # VIS uses its encoded value directly as both luminance and alpha.
    # The night side and outside-disk pixels are already 0 so they
    # contribute nothing to the composite without an explicit mask.
    vis_brightness = vis_encoded.astype(np.float32)
    vis_alpha = vis_encoded.astype(np.float32) / 255.0

    inv_vis_alpha = 1.0 - vis_alpha
    out_brightness = vis_brightness * vis_alpha + lw_brightness * inv_vis_alpha
    out_alpha = vis_alpha + lw_alpha * inv_vis_alpha

    rgba = _pack_rgba(out_brightness, out_alpha)
    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    if fmt == "webp":
        q = settings.webp_quality
        if q >= 100:
            img.save(buf, format="WEBP", lossless=True)
        else:
            img.save(buf, format="WEBP", quality=q)
    else:
        img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()

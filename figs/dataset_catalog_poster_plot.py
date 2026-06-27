#!/usr/bin/env python3
"""Poster-style dataset catalog from figs/table.csv — bit depth bands × sample rate grid."""

import argparse
import io
import sys
from functools import lru_cache
from os.path import dirname, join, realpath
from pathlib import Path as PathLib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from PIL import Image, ImageDraw, ImageFont

BIT_DEPTHS = [8, 16, 24]
BIT_COL = "b (Bits)"
FS_COL = "fs (kHz)"
CH_COL = "c (#Ch)"
DOMAIN_COL = "Domain"
DATASET_COL = "Dataset"

FIG_WIDTH = 10
FIGSIZE = (FIG_WIDTH, FIG_WIDTH / 2.2)
FONT_SIZE = 8
DOMAIN_EMOJI = {
    "Bioacoustics": "🦤",
    "Music": "🎶",
    "Speech": "🗣️",
    "SFX": "🔊",
}
# Noto draws the speech bubble with visual weight left of its trimmed bbox center.
DOMAIN_EMOJI_X_NUDGE_PX = {"Speech": 5}
GLYPH_PAD_X = 10
GLYPH_PAD_Y = 4
GLYPH_CHANNEL_PAD = 4
CHANNEL_LABEL_INSET_PX = 7
GLYPH_INNER_GAP = 2
EMOJI_DATASET_GAP_PX = 4
EMOJI_VERTICAL_FRACTION = 0.68
GLYPH_WIDTH_EXTRA_PX = 28
GLYPH_GAP = 0.10
ROW_GAP = 0.12
X_PAD = 0.08
BOTTOM_AXIS_CLEARANCE = 0.12
TOP_AXIS_CLEARANCE = 0.18
DOMAIN_LIGHTEN = 0.72
DOMAIN_DARKEN = 0.35
# Matplotlib tab10 defaults (C0–C3) — same family as table_poster_plot bar colors.
DOMAIN_BASE_COLORS = {
    "Bioacoustics": "#1f77b4",
    "Music": "#ff7f0e",
    "SFX": "#2ca02c",
    "Speech": "#d62728",
}
GLYPH_EDGE_WIDTH = 0.9
# Corner radius in typographic points — equal quarter-circles on screen (not data units).
VISUAL_CORNER_RADIUS_PT = 5
GLYPH_CORNER_ARC_PTS = 12
EMOJI_NATIVE_SIZE = 109
EMOJI_DISPLAY_PX = 20
APPLE_COLOR_EMOJI_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
APPLE_EMOJI_RENDER_SIZE = 160
NOTO_EMOJI_SIZES = (109, 137, 128, 96, 160, 64)

# Common install locations for Noto Color Emoji across Linux, macOS, and Windows.
EMOJI_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/usr/local/share/fonts/NotoColorEmoji.ttf",
    str(PathLib.home() / "Library/Fonts/NotoColorEmoji.ttf"),
    str(PathLib.home() / "Library/Fonts/NotoColorEmoji-Regular.ttf"),
    "/Library/Fonts/NotoColorEmoji.ttf",
    "/Library/Fonts/NotoColorEmoji-Regular.ttf",
    "/System/Library/Fonts/Supplemental/NotoColorEmoji.ttf",
    "/opt/homebrew/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
    "/opt/homebrew/Caskroom/font-noto-color-emoji/current/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/NotoColorEmoji.ttf",
    str(PathLib.home() / "AppData/Local/Microsoft/Windows/Fonts/NotoColorEmoji.ttf"),
)

EMOJI_FONT_SEARCH_DIRS = (
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    str(PathLib.home() / "Library/Fonts"),
    "/opt/homebrew/share/fonts",
    "/opt/homebrew/Caskroom/font-noto-color-emoji",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "C:/Windows/Fonts",
    str(PathLib.home() / "AppData/Local/Microsoft/Windows/Fonts"),
)

_EMOJI_FONT_TEST_GLYPH = "🎵"
_EMOJI_BACKEND: str | None = None  # "apple" or "noto"
_EMOJI_FONT_PATH: str | None = None
_NOTO_EMOJI_SIZE: int = EMOJI_NATIVE_SIZE


def _apple_emoji_available() -> bool:
    return sys.platform == "darwin" and PathLib(APPLE_COLOR_EMOJI_PATH).is_file()


def _bitmap_rep_to_rgba(rep, width: int, height: int) -> np.ndarray:
    """Read RGBA from an NSBitmapImageRep (fallback when PNG export fails)."""
    import AppKit

    bpr = int(rep.bytesPerRow())
    if int(rep.bitsPerPixel()) != 32 or int(rep.samplesPerPixel()) != 4:
        raise OSError(
            f"Unexpected bitmap format: bpp={rep.bitsPerPixel()}, spp={rep.samplesPerPixel()}"
        )
    buf = rep.bitmapData()
    if buf is None:
        raise OSError("NSBitmapImageRep has no bitmap data")
    row = bpr // 4
    pixels = np.frombuffer(buf, dtype=np.uint8).reshape((height, row, 4))[:, :width, :]
    rgba = pixels[:, :, [2, 1, 0, 3]].copy()  # BGRA -> RGBA
    return rgba[::-1, :, :]  # AppKit bitmap origin is bottom-left


def _apple_emoji_rgba_appkit(emoji: str, size: int = APPLE_EMOJI_RENDER_SIZE) -> np.ndarray:
    """Render via macOS AppKit — the reliable way to get Apple Color Emoji."""
    try:
        import AppKit
        import objc
        from Foundation import NSMakeRect
    except ImportError as err:
        raise ImportError(
            "Apple Color Emoji on macOS requires PyObjC:\n"
            "  pip install pyobjc-framework-Cocoa"
        ) from err

    with objc.autorelease_pool():
        font = AppKit.NSFont.fontWithName_size_("Apple Color Emoji", float(size))
        if font is None:
            raise OSError("Apple Color Emoji font not available on this system")

        attrs = AppKit.NSMutableDictionary.dictionary()
        attrs[AppKit.NSFontAttributeName] = font
        attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(emoji, attrs)
        text_size = attr_str.size()
        width = max(int(np.ceil(text_size.width)) + 4, 1)
        height = max(int(np.ceil(text_size.height)) + 4, 1)

        rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None,
            width,
            height,
            8,
            4,
            True,
            False,
            AppKit.NSCalibratedRGBColorSpace,
            0,
            0,
        )
        if rep is None:
            raise OSError("Failed to allocate NSBitmapImageRep for emoji rendering")

        AppKit.NSGraphicsContext.saveGraphicsState()
        ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
        AppKit.NSGraphicsContext.setCurrentContext_(ctx)

        AppKit.NSColor.clearColor().set()
        AppKit.NSRectFill(NSMakeRect(0, 0, width, height))
        attr_str.drawInRect_(NSMakeRect(0, 0, width, height))
        AppKit.NSGraphicsContext.restoreGraphicsState()

        png_type = getattr(AppKit, "NSBitmapImageFileTypePNG", AppKit.NSPNGFileType)
        png_data = rep.representationUsingType_properties_(png_type, {})
        if png_data is not None:
            data = bytes(png_data)
            if len(data) >= 8:
                return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))

        return _bitmap_rep_to_rgba(rep, width, height)


def validate_apple_emoji() -> None:
    if not _apple_emoji_available():
        raise ValueError("Apple Color Emoji is only available on macOS.")
    arr = _apple_emoji_rgba_appkit(_EMOJI_FONT_TEST_GLYPH)
    if arr[:, :, 3].max() < 16:
        raise ValueError("Apple Color Emoji did not render a visible glyph.")


def _noto_render_size(font_path: str) -> int:
    """Return a Pillow-compatible pixel size for a Noto Color Emoji file."""
    last_err: OSError | None = None
    for size in NOTO_EMOJI_SIZES:
        try:
            font = ImageFont.truetype(font_path, size)
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), _EMOJI_FONT_TEST_GLYPH, font=font, embedded_color=True)
            arr = np.asarray(image)
            if arr[:, :, 3].max() >= 16 and (arr[:, :, 3] > 16).sum() >= 30:
                return size
        except OSError as err:
            last_err = err
            if "SVG hooks" in str(err):
                raise ValueError(
                    f"Incompatible emoji font: {font_path}\n"
                    "Pillow cannot render this font (SVG/COLR). "
                    "On macOS use --emoji-backend apple instead."
                ) from err
    raise ValueError(
        f"Incompatible Noto emoji font: {font_path}\n"
        + (f"  {last_err}\n" if last_err else "")
        + "Could not render a test emoji at any supported size."
    )


def validate_noto_emoji_font(font_path: str) -> int:
    return _noto_render_size(font_path)


def discover_emoji_font_paths() -> list[str]:
    """Return candidate emoji font paths on this machine (not all are compatible)."""
    seen: set[str] = set()
    paths: list[str] = []

    def add(path: PathLib) -> None:
        if not path.is_file():
            return
        resolved = str(path.expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    for candidate in EMOJI_FONT_CANDIDATES:
        add(PathLib(candidate))

    brew_cask = PathLib("/opt/homebrew/Caskroom/font-noto-color-emoji")
    if brew_cask.is_dir():
        for path in sorted(brew_cask.glob("**/NotoColorEmoji*.ttf")):
            add(path)

    name_patterns = ("*Color*Emoji*", "*color*emoji*", "*NotoColorEmoji*")
    for root in EMOJI_FONT_SEARCH_DIRS:
        root_path = PathLib(root).expanduser()
        if not root_path.is_dir():
            continue
        for pattern in name_patterns:
            for path in root_path.rglob(pattern):
                if path.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                    add(path)

    return paths


def find_viable_noto_font() -> tuple[str, int]:
    for path in discover_emoji_font_paths():
        try:
            return path, validate_noto_emoji_font(path)
        except ValueError:
            continue

    raise FileNotFoundError(
        "No Pillow-compatible Noto Color Emoji found.\n"
        "  macOS: use --emoji-backend apple (default), or brew install --cask font-noto-color-emoji\n"
        "  Debian/Ubuntu: sudo apt install fonts-noto-color-emoji\n"
        "  Fedora: sudo dnf install google-noto-emoji-fonts"
    )


def list_emoji_fonts() -> None:
    print("Emoji rendering options on this machine:\n")

    if _apple_emoji_available():
        try:
            validate_apple_emoji()
            print(f"  OK   Apple Color Emoji (AppKit) — {APPLE_COLOR_EMOJI_PATH}")
            print("       Use: --emoji-backend apple  (default on macOS)")
        except (ImportError, ValueError, OSError) as err:
            print(f"  skip Apple Color Emoji (AppKit)")
            for line in str(err).split("\n"):
                print(f"       {line}")
    else:
        print("  —    Apple Color Emoji (macOS only)")

    print()
    print("Noto Color Emoji (Pillow):")
    found = discover_emoji_font_paths()
    if not found:
        print("  No Noto font files found.")
    else:
        any_ok = False
        for path in found:
            try:
                size = validate_noto_emoji_font(path)
                print(f"  OK   {path}  (size {size})")
                any_ok = True
            except ValueError as err:
                print(f"  skip {path}")
                print(f"       {str(err).splitlines()[0]}")
        if not any_ok:
            print("  No Pillow-compatible Noto fonts found.")

    print(
        "\nApple Color Emoji cannot be rendered by Pillow (SVG hooks error).\n"
        "On Mac, use --emoji-backend apple and: pip install pyobjc-framework-Cocoa"
    )


def configure_emoji_font(
    emoji_font_path: str | None = None,
    emoji_backend: str = "auto",
) -> str:
    global _EMOJI_BACKEND, _EMOJI_FONT_PATH, _NOTO_EMOJI_SIZE

    if emoji_font_path and emoji_backend == "apple":
        raise ValueError("--emoji-font-path cannot be used with --emoji-backend apple.")

    if emoji_backend == "apple" or (emoji_backend == "auto" and _apple_emoji_available()):
        try:
            validate_apple_emoji()
            _EMOJI_BACKEND = "apple"
            _EMOJI_FONT_PATH = APPLE_COLOR_EMOJI_PATH
            _emoji_rgba.cache_clear()
            _emoji_centroid_offset.cache_clear()
            _noto_emoji_rgba.cache_clear()
            return f"Apple Color Emoji (AppKit, {APPLE_COLOR_EMOJI_PATH})"
        except (ImportError, ValueError, OSError):
            if emoji_backend == "apple":
                raise
            # auto: fall through to Noto on Mac without PyObjC

    if emoji_font_path:
        path = str(PathLib(emoji_font_path).expanduser().resolve())
        if not PathLib(path).is_file():
            raise FileNotFoundError(f"Emoji font not found: {emoji_font_path}")
        size = validate_noto_emoji_font(path)
    else:
        path, size = find_viable_noto_font()

    _EMOJI_BACKEND = "noto"
    _EMOJI_FONT_PATH = path
    _NOTO_EMOJI_SIZE = size
    _emoji_rgba.cache_clear()
    _emoji_centroid_offset.cache_clear()
    _noto_emoji_rgba.cache_clear()
    return f"Noto Color Emoji ({path}, size {size})"


def _trim_rgba(arr: np.ndarray) -> np.ndarray:
    """Drop transparent margins so wide emoji canvases center visually."""
    alpha = arr[:, :, 3]
    if alpha.max() == 0:
        return arr
    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


@lru_cache(maxsize=16)
def _noto_emoji_rgba(emoji: str) -> np.ndarray:
    if _EMOJI_FONT_PATH is None:
        raise RuntimeError("Call configure_emoji_font() before rendering emojis.")
    font = ImageFont.truetype(_EMOJI_FONT_PATH, _NOTO_EMOJI_SIZE)
    bbox = font.getbbox(emoji)
    width = max(bbox[2] - bbox[0], 1)
    height = max(bbox[3] - bbox[1], 1)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((-bbox[0], -bbox[1]), emoji, font=font, embedded_color=True)
    return np.asarray(image)


@lru_cache(maxsize=16)
def _emoji_rgba(emoji: str) -> np.ndarray:
    if _EMOJI_BACKEND == "apple":
        return _trim_rgba(_apple_emoji_rgba_appkit(emoji))
    return _trim_rgba(_noto_emoji_rgba(emoji))


def _emoji_zoom(emoji: str) -> float:
    return EMOJI_DISPLAY_PX / _emoji_rgba(emoji).shape[0]


def _emoji_size_px(ax, renderer, emoji: str) -> tuple[float, float]:
    arr = _emoji_rgba(emoji)
    zoom = _emoji_zoom(emoji)
    return arr.shape[1] * zoom, arr.shape[0] * zoom


@lru_cache(maxsize=16)
def _emoji_centroid_offset(emoji: str) -> tuple[float, float]:
    """Pixel offset from image center to visual centroid (for centering lopsided glyphs)."""
    arr = _emoji_rgba(emoji)
    alpha = arr[:, :, 3].astype(float)
    total = alpha.sum()
    if total == 0:
        return 0.0, 0.0
    ys, xs = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    cx = (xs * alpha).sum() / total
    cy = (ys * alpha).sum() / total
    return cx - arr.shape[1] / 2, cy - arr.shape[0] / 2


def _text_bbox_px(
    ax, renderer, text: str, *, fontweight: str = "normal", fontsize: float = FONT_SIZE
) -> tuple[float, float]:
    text_obj = ax.text(
        0, 0, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight
    )
    bbox = text_obj.get_window_extent(renderer)
    text_obj.remove()
    return bbox.width, bbox.height


def _display_px_per_data_unit(ax, x: float, y: float) -> tuple[float, float]:
    """Display pixels per 1 data unit in x and y at (x, y)."""
    origin = np.array(ax.transData.transform((x, y)))
    px_x = np.array(ax.transData.transform((x + 1.0, y)))
    px_y = np.array(ax.transData.transform((x, y + 1.0)))
    return float(np.linalg.norm(px_x - origin)), float(np.linalg.norm(px_y - origin))


def _pad_data(ax, x: float, y: float, pad_x_px: float, pad_y_px: float) -> tuple[float, float]:
    sx, sy = _display_px_per_data_unit(ax, x, y)
    return pad_x_px / sx, pad_y_px / sy


def _glyph_content_metrics(
    ax, renderer, domain: str, dataset: str, ch: str
) -> tuple[float, float]:
    """Return content width/height in display pixels for the new glyph layout."""
    emoji_w, emoji_h = _emoji_size_px(ax, renderer, DOMAIN_EMOJI[domain])
    dataset_w, dataset_h = _text_bbox_px(ax, renderer, dataset, fontweight="bold")
    ch_w, ch_h = _text_bbox_px(ax, renderer, ch)

    width_px = (
        max(emoji_w, dataset_w, ch_w + emoji_w * 0.35)
        + 2 * GLYPH_PAD_X
        + GLYPH_WIDTH_EXTRA_PX
    )
    height_px = (
        GLYPH_CHANNEL_PAD
        + ch_h
        + GLYPH_INNER_GAP
        + emoji_h
        + GLYPH_INNER_GAP
        + dataset_h
        + GLYPH_PAD_Y
    )
    return width_px, height_px


def _draw_emoji(
    ax,
    domain: str,
    emoji_x: float,
    emoji_y: float,
    clip_patch: PathPatch,
) -> None:
    emoji = DOMAIN_EMOJI[domain]
    arr = _emoji_rgba(emoji)
    zoom = _emoji_zoom(emoji)
    emoji_box = AnnotationBbox(
        OffsetImage(arr, zoom=zoom, interpolation="nearest"),
        (emoji_x, emoji_y),
        frameon=False,
        pad=0,
        box_alignment=(0.5, 0.5),
        zorder=4,
    )
    emoji_box.set_clip_path(clip_patch)
    ax.add_artist(emoji_box)


def _draw_glyph_content(
    ax,
    x: float,
    y: float,
    domain: str,
    dataset: str,
    ch: str,
    color: tuple,
    glyph_w: float,
    glyph_h: float,
    clip_patch: PathPatch,
    fig,
) -> None:
    emoji = DOMAIN_EMOJI[domain]
    _, pad_y = _pad_data(ax, x, y, GLYPH_PAD_X, GLYPH_PAD_Y)
    layout_channel_pad = _pad_data(ax, x, y, GLYPH_CHANNEL_PAD, GLYPH_CHANNEL_PAD)
    label_channel_pad = _pad_data(
        ax,
        x,
        y,
        GLYPH_CHANNEL_PAD + CHANNEL_LABEL_INSET_PX,
        GLYPH_CHANNEL_PAD + CHANNEL_LABEL_INSET_PX,
    )

    x1 = x + glyph_w / 2
    y0 = y - glyph_h / 2
    y1 = y + glyph_h / 2

    renderer = fig.canvas.get_renderer()
    _, ch_h_px = _text_bbox_px(ax, renderer, ch)
    _, dataset_h_px = _text_bbox_px(ax, renderer, dataset, fontweight="bold")
    channel_h = _pad_data(ax, x, y, 0, ch_h_px)[1]
    dataset_h = _pad_data(ax, x, y, 0, dataset_h_px)[1]

    ch_x = x1 - label_channel_pad[0]
    ch_y = y1 - label_channel_pad[1]
    channel_bottom = y1 - layout_channel_pad[1] - channel_h
    dataset_y = y0 + pad_y
    dataset_top = dataset_y + dataset_h
    _, emoji_h_px = _emoji_size_px(ax, renderer, emoji)
    emoji_h = _pad_data(ax, x, y, 0, emoji_h_px)[1]
    emoji_gap = _pad_data(ax, x, y, 0, EMOJI_DATASET_GAP_PX)[1]
    available = channel_bottom - dataset_top - emoji_gap - emoji_h
    emoji_y = dataset_top + emoji_gap + emoji_h / 2 + max(available, 0) * EMOJI_VERTICAL_FRACTION

    sx, _ = _display_px_per_data_unit(ax, x, y)
    centroid_x_px, _ = _emoji_centroid_offset(emoji)
    emoji_x = x - (centroid_x_px * _emoji_zoom(emoji)) / sx
    emoji_x += DOMAIN_EMOJI_X_NUDGE_PX.get(domain, 0) / sx

    _draw_emoji(ax, domain, emoji_x, emoji_y, clip_patch)

    ax.text(
        x,
        dataset_y,
        dataset,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=color,
        clip_on=False,
        zorder=4,
    )
    ax.text(
        ch_x,
        ch_y,
        ch,
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="normal",
        color=color,
        clip_on=False,
        zorder=4,
    )


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="dataset_catalog_poster_plot",
        description="Plot dataset catalog poster from table.csv",
    )
    default_dir = dirname(realpath(__file__))
    parser.add_argument(
        "--input_filepath",
        type=str,
        default=join(default_dir, "table.csv"),
        help="Absolute filepath to the input CSV file.",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default=join(default_dir, "dataset_catalog_poster_plot.pdf"),
        help="Absolute filepath to the output PDF file.",
    )
    parser.add_argument(
        "--emoji-font-path",
        type=str,
        default=None,
        help="Noto Color Emoji .ttf path (only with --emoji-backend noto).",
    )
    parser.add_argument(
        "--emoji-backend",
        choices=("auto", "apple", "noto"),
        default="auto",
        help="Emoji renderer: auto (Apple on macOS, else Noto), apple, or noto.",
    )
    parser.add_argument(
        "--list-emoji-fonts",
        action="store_true",
        help="List emoji rendering options on this machine, then exit.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def load_data(input_filepath: str) -> pd.DataFrame:
    return pd.read_csv(input_filepath)


def channel_suffix(n_ch: int) -> str:
    return "M" if n_ch == 1 else "S"


def format_fs_label(rate: float) -> str:
    return f"{rate:.2f}"


def _lighten(rgb: tuple, amount: float) -> tuple:
    c = np.array(mcolors.to_rgb(rgb))
    return tuple(c + (1 - c) * amount)


def _darken(rgb: tuple, amount: float) -> tuple:
    c = np.array(mcolors.to_rgb(rgb))
    return tuple(c * (1 - amount))


def domain_palette(df: pd.DataFrame) -> dict[str, dict[str, tuple]]:
    """Per domain: base color plus light (fill) and dark (text/border) variants."""
    domains = sorted(df[DOMAIN_COL].unique())
    return {
        domain: {
            "base": mcolors.to_rgb(DOMAIN_BASE_COLORS[domain]),
            "light": _lighten(DOMAIN_BASE_COLORS[domain], DOMAIN_LIGHTEN),
            "dark": _darken(DOMAIN_BASE_COLORS[domain], DOMAIN_DARKEN),
        }
        for domain in domains
    }



def _px_size_to_data(ax, anchor: tuple[float, float], width_px: float, height_px: float) -> tuple[float, float]:
    """Convert a display-pixel width/height at anchor to data-axis extents."""
    inv = ax.transData.inverted()
    cx, cy = ax.transData.transform(anchor)
    (x0, y0) = inv.transform((cx - width_px / 2, cy - height_px / 2))
    (x1, y1) = inv.transform((cx + width_px / 2, cy + height_px / 2))
    return x1 - x0, y1 - y0


def measure_glyph_size(fig, ax, df: pd.DataFrame) -> tuple[float, float]:
    """Fixed glyph size from the widest/tallest label layout across all datasets."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    anchor = (5.0, 5.0)

    max_w_px = 0.0
    max_h_px = 0.0
    for _, row in df.iterrows():
        w_px, h_px = _glyph_content_metrics(
            ax,
            renderer,
            row[DOMAIN_COL],
            row[DATASET_COL],
            channel_suffix(int(row[CH_COL])),
        )
        max_w_px = max(max_w_px, w_px)
        max_h_px = max(max_h_px, h_px)

    return _px_size_to_data(ax, anchor, max_w_px, max_h_px)


def band_spans(df: pd.DataFrame, glyph_w: float, gap: float) -> dict[int, tuple[float, float]]:
    """Band width = widest packed row (n glyphs + gaps) within each bit depth."""
    spans: dict[int, tuple[float, float]] = {}
    cursor = 0.0
    for bit in BIT_DEPTHS:
        band_df = df[df[BIT_COL] == bit]
        band_width = glyph_w + 2 * gap
        if not band_df.empty:
            for _, group in band_df.groupby(FS_COL):
                n = len(group)
                band_width = max(band_width, n * glyph_w + (n + 1) * gap)
        spans[bit] = (cursor, cursor + band_width)
        cursor += band_width
    return spans


def fs_y_positions(df: pd.DataFrame, row_pitch: float) -> dict[float, float]:
    rates = sorted(df[FS_COL].unique())
    return {rate: i * row_pitch for i, rate in enumerate(rates)}


def glyph_x(
    x_left: float,
    x_right: float,
    index: int,
    n: int,
    glyph_w: float,
    gap: float,
) -> float:
    band_width = x_right - x_left
    row_width = n * glyph_w + (n + 1) * gap
    row_start = x_left + (band_width - row_width) / 2
    return row_start + gap + glyph_w / 2 + index * (glyph_w + gap)


def glyph_corner_radii(
    ax, x: float, y: float, glyph_w: float, glyph_h: float, fig
) -> tuple[float, float]:
    """Return rx, ry in data units so corners are circular quarter-circles on screen."""
    r_px = VISUAL_CORNER_RADIUS_PT * fig.dpi / 72.0
    sx, sy = _display_px_per_data_unit(ax, x, y)
    rx = min(r_px / sx, glyph_w / 2)
    ry = min(r_px / sy, glyph_h / 2)
    return rx, ry


def _append_ellipse_arc(
    verts: list,
    codes: list,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    t0: float,
    t1: float,
    *,
    skip_first: bool,
) -> None:
    for i, t in enumerate(np.linspace(t0, t1, GLYPH_CORNER_ARC_PTS)):
        if skip_first and i == 0:
            continue
        verts.append((cx + rx * np.cos(t), cy + ry * np.sin(t)))
        codes.append(Path.LINETO)


def rounded_rect_path(x0: float, y0: float, w: float, h: float, rx: float, ry: float) -> Path:
    """Rounded rect; rx/ry in data units chosen so corners appear as equal quarter-circles."""
    rx = min(rx, w / 2)
    ry = min(ry, h / 2)
    x1, y1 = x0 + w, y0 + h
    if rx <= 0 or ry <= 0:
        return Path(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        )

    verts: list[tuple[float, float]] = [(x0 + rx, y0)]
    codes: list[int] = [Path.MOVETO]

    verts.append((x1 - rx, y0))
    codes.append(Path.LINETO)
    _append_ellipse_arc(verts, codes, x1 - rx, y0 + ry, rx, ry, -np.pi / 2, 0, skip_first=True)

    verts.append((x1, y1 - ry))
    codes.append(Path.LINETO)
    _append_ellipse_arc(verts, codes, x1 - rx, y1 - ry, rx, ry, 0, np.pi / 2, skip_first=True)

    verts.append((x0 + rx, y1))
    codes.append(Path.LINETO)
    _append_ellipse_arc(verts, codes, x0 + rx, y1 - ry, rx, ry, np.pi / 2, np.pi, skip_first=True)

    verts.append((x0, y0 + ry))
    codes.append(Path.LINETO)
    _append_ellipse_arc(verts, codes, x0 + rx, y0 + ry, rx, ry, np.pi, 3 * np.pi / 2, skip_first=True)

    codes.append(Path.CLOSEPOLY)
    verts.append((0.0, 0.0))
    return Path(verts, codes)


def draw_glyph(
    ax,
    x: float,
    y: float,
    domain: str,
    dataset: str,
    ch: str,
    domain_style: dict[str, tuple],
    glyph_w: float,
    glyph_h: float,
) -> None:
    x0 = x - glyph_w / 2
    y0 = y - glyph_h / 2
    rx, ry = glyph_corner_radii(ax, x, y, glyph_w, glyph_h, ax.figure)
    rect = PathPatch(
        rounded_rect_path(x0, y0, glyph_w, glyph_h, rx, ry),
        facecolor=domain_style["light"],
        edgecolor=domain_style["dark"],
        linewidth=GLYPH_EDGE_WIDTH,
        joinstyle="miter",
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(rect)
    _draw_glyph_content(
        ax,
        x,
        y,
        domain,
        dataset,
        ch,
        domain_style["dark"],
        glyph_w,
        glyph_h,
        rect,
        ax.figure,
    )


def domain_legend_handles_labels(palette: dict[str, dict[str, tuple]]) -> tuple[list, list]:
    domains = sorted(palette)
    handles = [
        Patch(
            facecolor=palette[d]["light"],
            edgecolor=palette[d]["dark"],
            linewidth=GLYPH_EDGE_WIDTH,
            label=d,
        )
        for d in domains
    ]
    return handles, domains


def style_axes(
    ax,
    spans: dict[int, tuple[float, float]],
    fs_to_y: dict[float, float],
    row_pitch: float,
    glyph_h: float,
) -> None:
    total_width = spans[BIT_DEPTHS[-1]][1]
    ax.set_xlim(-X_PAD, total_width + X_PAD)

    y_max = max(fs_to_y.values()) if fs_to_y else 0
    y_min_row = min(fs_to_y.values()) if fs_to_y else 0
    bottom = y_min_row - glyph_h / 2 - row_pitch * BOTTOM_AXIS_CLEARANCE
    top = y_max + glyph_h / 2 + row_pitch * TOP_AXIS_CLEARANCE
    ax.set_ylim(bottom, top)

    band_centers = [(spans[b][0] + spans[b][1]) / 2 for b in BIT_DEPTHS]
    boundaries = [spans[b][1] for b in BIT_DEPTHS[:-1]]
    ax.set_xticks(band_centers)
    ax.set_xticklabels([f"{b}-bit" for b in BIT_DEPTHS])
    ax.set_xticks(boundaries, minor=True)
    spine_lw = ax.spines["bottom"].get_linewidth()
    ax.tick_params(axis="x", which="major", length=0, pad=7, labelbottom=True)
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        bottom=True,
        length=7,
        width=spine_lw,
    )

    y_ticks = [fs_to_y[r] for r in sorted(fs_to_y)]
    y_labels = [format_fs_label(r) for r in sorted(fs_to_y)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Sample Rate (kHz)")
    ax.tick_params(axis="y", which="major", direction="out", left=True)

    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, axis="y")


def plot_catalog(df: pd.DataFrame, ax, fig) -> tuple[list, list]:
    palette = domain_palette(df)
    glyph_w, glyph_h = measure_glyph_size(fig, ax, df)
    gap = GLYPH_GAP
    row_pitch = glyph_h + ROW_GAP

    spans = band_spans(df, glyph_w, gap)
    fs_to_y = fs_y_positions(df, row_pitch)
    style_axes(ax, spans, fs_to_y, row_pitch, glyph_h)
    fig.canvas.draw()

    for bit in BIT_DEPTHS:
        x_left, x_right = spans[bit]
        band_df = df[df[BIT_COL] == bit]

        for fs, group in band_df.groupby(FS_COL, sort=True):
            y = fs_to_y[fs]
            n = len(group)
            for i, (_, row) in enumerate(group.iterrows()):
                x = glyph_x(x_left, x_right, i, n, glyph_w, gap)
                ch = channel_suffix(int(row[CH_COL]))
                draw_glyph(
                    ax,
                    x,
                    y,
                    row[DOMAIN_COL],
                    row[DATASET_COL],
                    ch,
                    palette[row[DOMAIN_COL]],
                    glyph_w,
                    glyph_h,
                )

    return domain_legend_handles_labels(palette)


def main() -> None:
    args = parse_args()
    if args.list_emoji_fonts:
        list_emoji_fonts()
        return

    emoji_font = configure_emoji_font(args.emoji_font_path, args.emoji_backend)
    df = load_data(args.input_filepath)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_alpha(0)

    legend_handles, legend_labels = plot_catalog(df, ax, fig)
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        title="Domain",
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92)
    plt.savefig(
        args.output_filepath, dpi=300, bbox_inches="tight", pad_inches=0.08, transparent=True
    )
    print(f"Using emoji font: {emoji_font}")
    print(f"Saved plot to {args.output_filepath}.")


if __name__ == "__main__":
    main()

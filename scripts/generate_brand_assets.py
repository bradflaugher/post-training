from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

BLACK = (8, 5, 8, 255)
INK_BLACK = (0, 0, 0, 255)
CRIMSON = (230, 0, 38, 255)
DARK_RED = (130, 0, 20, 255)
ACID_GREEN = (57, 255, 20, 255)
BONE = (244, 238, 224, 255)
CORTEX_PINK = (255, 77, 121, 255)
TRANSPARENT = (0, 0, 0, 0)

SVG_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" role="img" aria-labelledby="title desc">
  <title id="title">POST-TRAINING icon</title>
  <desc id="desc">A comic-book bisected cybernetic brain dripping crimson ink with an acid-green scalpel slash.</desc>
  <rect width="1024" height="1024" rx="220" fill="#080508"/>
  <!-- Blood splatter background -->
  <circle cx="240" cy="240" r="110" fill="#820014"/>
  <circle cx="790" cy="760" r="140" fill="#820014"/>
  <circle cx="780" cy="250" r="65" fill="#e60026"/>
  <circle cx="230" cy="770" r="75" fill="#e60026"/>
  <!-- Left & Right Brain Hemispheres with heavy black comic ink outline -->
  <path d="M475 210 C310 200 185 320 190 485 C170 610 260 760 445 775 L475 680 Z" fill="#ff4d79" stroke="#000" stroke-width="44" stroke-linejoin="round"/>
  <path d="M549 235 C714 225 839 345 834 510 C854 635 764 785 579 800 L549 705 Z" fill="#e60026" stroke="#000" stroke-width="44" stroke-linejoin="round"/>
  <!-- Brain gyri ink squiggles -->
  <path d="M270 430 Q360 370 420 450 M260 560 Q350 620 430 540 M600 410 Q690 360 750 450 M595 575 Q685 635 760 555" fill="none" stroke="#080508" stroke-width="34" stroke-linecap="round"/>
  <!-- Acid-green scalpel / logit slash cleaving the corpus callosum -->
  <path d="M625 110 L395 914" fill="none" stroke="#000" stroke-width="96" stroke-linecap="round"/>
  <path d="M625 110 L395 914" fill="none" stroke="#39ff14" stroke-width="48" stroke-linecap="round"/>
</svg>
"""

SVG_MARK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" role="img" aria-labelledby="title desc">
  <title id="title">POST-TRAINING mark</title>
  <desc id="desc">Bisected brain and scalpel slash mark.</desc>
  <path d="M475 210 C310 200 185 320 190 485 C170 610 260 760 445 775 L475 680 Z" fill="currentColor"/>
  <path d="M549 235 C714 225 839 345 834 510 C854 635 764 785 579 800 L549 705 Z" fill="currentColor"/>
  <path d="M625 110 L395 914" fill="none" stroke="currentColor" stroke-width="72" stroke-linecap="round"/>
</svg>
"""

SVG_MASK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <path d="M625 110 L395 914" fill="none" stroke="#000" stroke-width="124" stroke-linecap="round"/>
</svg>
"""


def save_svg_assets() -> None:
    (ROOT / "favicon.svg").write_text(SVG_ICON, encoding="utf-8")
    (ROOT / "post-training-icon.svg").write_text(SVG_ICON, encoding="utf-8")
    (ROOT / "post-training-mark.svg").write_text(SVG_MARK, encoding="utf-8")
    (ROOT / "safari-pinned-tab.svg").write_text(SVG_MASK, encoding="utf-8")


def draw_emblem(draw: ImageDraw.ImageDraw, size: int, scale: float = 1.0, mono_fg=None) -> None:
    cx = cy = size / 2

    def sc(x: float, y: float) -> tuple[float, float]:
        return (cx + (x * size - cx) * scale, cy + (y * size - cy) * scale)

    if mono_fg is not None:
        # Clean monochrome slash + brain lobes
        l_box = (*sc(0.18, 0.22), *sc(0.47, 0.78))
        r_box = (*sc(0.53, 0.24), *sc(0.82, 0.80))
        draw.pieslice(l_box, start=90, end=270, fill=mono_fg)
        draw.pieslice(r_box, start=270, end=90, fill=mono_fg)
        x1, y1 = sc(0.61, 0.12)
        x2, y2 = sc(0.39, 0.88)
        w = max(2, int(size * 0.09 * scale))
        draw.line((x1, y1, x2, y2), fill=mono_fg, width=w)
        return

    # Crimson splatter dots
    for sx, sy, sr, col in [
        (0.24, 0.24, 0.10, DARK_RED),
        (0.77, 0.75, 0.13, DARK_RED),
        (0.76, 0.24, 0.06, CRIMSON),
        (0.22, 0.76, 0.07, CRIMSON),
    ]:
        px, py = sc(sx, sy)
        pr = size * sr * scale
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=col)

    # Left hemisphere (Cortex Pink) & Right hemisphere (Crimson)
    l_box = (*sc(0.16, 0.20), *sc(0.72, 0.78))
    r_box = (*sc(0.28, 0.22), *sc(0.84, 0.80))
    outline_w = max(1, int(size * 0.045 * scale))
    draw.pieslice(l_box, start=100, end=265, fill=CORTEX_PINK, outline=INK_BLACK, width=outline_w)
    draw.pieslice(r_box, start=280, end=85, fill=CRIMSON, outline=INK_BLACK, width=outline_w)

    # Acid green scalpel slash right down the middle
    x1, y1 = sc(0.62, 0.11)
    x2, y2 = sc(0.38, 0.89)
    outer_w = max(3, int(size * 0.11 * scale))
    inner_w = max(2, int(size * 0.055 * scale))
    draw.line((x1, y1, x2, y2), fill=INK_BLACK, width=outer_w)
    draw.line((x1, y1, x2, y2), fill=ACID_GREEN, width=inner_w)
    r = inner_w / 2
    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=ACID_GREEN)
    draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), fill=ACID_GREEN)


def icon(size: int, bg=BLACK, radius_ratio: float = 0.21875, slash_scale: float = 1.0, mono_fg=None) -> Image.Image:
    radius = int(size * radius_ratio)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, size, size), radius=radius, fill=255)
    img = Image.new("RGBA", (size, size), bg)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    draw_emblem(draw, size, scale=slash_scale, mono_fg=mono_fg)
    return img


def maskable_icon(size: int, bg=BLACK) -> Image.Image:
    img = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)
    draw_emblem(draw, size, scale=0.82)
    return img


def mono_icon(size: int, fg=BONE) -> Image.Image:
    img = Image.new("RGBA", (size, size), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    draw_emblem(draw, size, scale=0.82, mono_fg=fg)
    return img


def find_font(size: int, weight: str = "regular"):
    candidates = [
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    if weight == "bold":
        candidates = [
            "/usr/share/fonts/TTF/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ] + candidates
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_social(width: int, height: int, out: str) -> None:
    img = Image.new("RGBA", (width, height), BLACK)
    draw = ImageDraw.Draw(img)

    # Heavy comic halftone / slash lines in background
    for i in range(14):
        x = int(width * (i / 12.0))
        draw.line((x + 220, -40, x - 120, height + 40), fill=(130, 0, 20, 45), width=6)

    # Comic border frame
    draw.rectangle((18, 18, width - 18, height - 18), outline=CRIMSON, width=6)
    draw.rectangle((30, 30, width - 30, height - 30), outline=ACID_GREEN, width=2)

    block = int(min(width, height) * 0.38)
    bx = int(width * 0.08)
    by = int((height - block) / 2)
    badge = icon(block, bg=BLACK, radius_ratio=0.22, slash_scale=0.95)
    img.alpha_composite(badge, (bx, by))

    title_font = find_font(int(height * 0.13), "bold")
    desc_font = find_font(int(height * 0.052), "bold")
    small_font = find_font(int(height * 0.036), "bold")
    tx = bx + block + int(width * 0.05)

    draw.text((tx + 4, int(height * 0.25) + 4), "POST-TRAINING", font=title_font, fill=CRIMSON)
    draw.text((tx, int(height * 0.25)), "POST-TRAINING", font=title_font, fill=BONE)
    draw.text((tx + 4, int(height * 0.48)), "NEURAL VIVISECTION WORD PUZZLE", font=desc_font, fill=ACID_GREEN)
    draw.text(
        (tx + 4, int(height * 0.62)), "16 TOKENS · 4 LOBES · ON-DEVICE LLM SURGERY", font=small_font, fill=CORTEX_PINK
    )

    img.convert("RGB").save(ROOT / out, optimize=True, quality=94)


def generate() -> None:
    save_svg_assets()

    for size in [16, 32, 48, 96, 144, 180, 192, 256, 384, 512, 1024]:
        if size in [16, 32, 48, 96, 144]:
            icon(size).save(ROOT / f"favicon-{size}x{size}.png")
        icon(size).save(ROOT / f"icon-{size}.png")

    for size in [120, 152, 167, 180, 192, 512, 1024]:
        icon(size).save(ROOT / f"apple-touch-icon-{size}x{size}.png")
    icon(180).save(ROOT / "apple-touch-icon.png")
    icon(180, bg=BONE).save(ROOT / "apple-touch-icon-light.png")
    maskable_icon(180, bg=BLACK).save(ROOT / "apple-touch-icon-dark.png")
    mono_icon(180, fg=BONE).save(ROOT / "apple-touch-icon-tinted.png")

    for size in [192, 512, 1024]:
        maskable_icon(size).save(ROOT / f"icon-maskable-{size}.png")
        mono_icon(size, fg=BONE).save(ROOT / f"icon-monochrome-{size}.png")

    icon(70).save(ROOT / "mstile-70x70.png")
    icon(150).save(ROOT / "mstile-150x150.png")
    icon(310).save(ROOT / "mstile-310x310.png")
    tile = Image.new("RGBA", (310, 150), BLACK)
    tile.alpha_composite(icon(112), (99, 19))
    tile.save(ROOT / "mstile-310x150.png")

    ico_sizes = [16, 32, 48, 64, 128, 256]
    base = icon(256)
    base.save(ROOT / "favicon.ico", sizes=[(s, s) for s in ico_sizes])

    make_social(1200, 630, "og-image.png")
    make_social(1200, 628, "twitter-card.png")
    make_social(1200, 1200, "og-image-square.png")


if __name__ == "__main__":
    generate()

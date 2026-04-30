"""
generate_soil_images.py
=======================
Generates 4000+ realistic synthetic soil images across 10 soil classes.
Each image uses procedural texture generation to mimic real soil photography:
  - Perlin-like noise for grain texture
  - Color variation per soil class
  - Lighting / shadow simulation
  - Moisture variation
  - Stone, pebble, and debris inclusions
  - Crack patterns for dry soils
  - Root fiber textures for organic soils

Soil Classes (400+ images each = 4000+ total):
  1. Black Soil       6. Laterite Soil
  2. Red Soil         7. Loamy Soil
  3. Alluvial Soil    8. Peat Soil
  4. Sandy Soil       9. Chalky Soil
  5. Clay Soil       10. Silt Soil
"""

import os
import sys
import random
import time
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from typing import Tuple, List

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "/home/claude/soil_dataset"
IMG_SIZE     = (224, 224)
IMAGES_PER_CLASS = 420   # 420 × 10 = 4200 total
SEED_BASE    = 1337

# ── Soil Color Palettes ────────────────────────────────────────────────────────
# Each soil has: (base_color_rgb, variation_range, secondary_colors, texture_style)
SOIL_PROFILES = {
    "black_soil": {
        "base":       (28, 20, 15),
        "variation":  18,
        "secondary":  [(35, 28, 22), (22, 15, 10), (40, 32, 25), (18, 12, 8)],
        "moisture_shift": (8, 12, 10),    # color shift when wet
        "texture":    "fine_clay",
        "has_cracks": True,
        "crack_color": (15, 10, 8),
        "pebble_chance": 0.10,
        "pebble_color": (60, 55, 50),
        "root_chance":  0.15,
    },
    "red_soil": {
        "base":       (160, 60, 35),
        "variation":  30,
        "secondary":  [(180, 75, 40), (140, 50, 28), (170, 90, 55), (130, 45, 22)],
        "moisture_shift": (12, 8, 5),
        "texture":    "granular",
        "has_cracks": False,
        "crack_color": (120, 45, 20),
        "pebble_chance": 0.25,
        "pebble_color": (145, 55, 30),
        "root_chance":  0.20,
    },
    "alluvial_soil": {
        "base":       (175, 148, 110),
        "variation":  28,
        "secondary":  [(190, 162, 122), (158, 132, 95), (182, 155, 118), (165, 138, 100)],
        "moisture_shift": (10, 8, 6),
        "texture":    "layered",
        "has_cracks": False,
        "crack_color": (150, 125, 90),
        "pebble_chance": 0.30,
        "pebble_color": (130, 110, 85),
        "root_chance":  0.25,
    },
    "sandy_soil": {
        "base":       (210, 185, 140),
        "variation":  35,
        "secondary":  [(225, 200, 155), (195, 170, 125), (218, 190, 148), (200, 178, 132)],
        "moisture_shift": (5, 4, 3),
        "texture":    "coarse_grain",
        "has_cracks": False,
        "crack_color": (190, 165, 120),
        "pebble_chance": 0.45,
        "pebble_color": (195, 175, 135),
        "root_chance":  0.10,
    },
    "clay_soil": {
        "base":       (148, 115, 82),
        "variation":  22,
        "secondary":  [(162, 128, 92), (132, 100, 70), (155, 120, 88), (140, 108, 76)],
        "moisture_shift": (15, 12, 8),
        "texture":    "smooth_clay",
        "has_cracks": True,
        "crack_color": (115, 88, 60),
        "pebble_chance": 0.08,
        "pebble_color": (120, 95, 68),
        "root_chance":  0.12,
    },
    "laterite_soil": {
        "base":       (185, 88, 42),
        "variation":  28,
        "secondary":  [(200, 100, 52), (168, 75, 35), (192, 92, 48), (175, 82, 38)],
        "moisture_shift": (10, 6, 4),
        "texture":    "rough_granular",
        "has_cracks": False,
        "crack_color": (155, 68, 30),
        "pebble_chance": 0.35,
        "pebble_color": (160, 72, 35),
        "root_chance":  0.18,
    },
    "loamy_soil": {
        "base":       (118, 88, 55),
        "variation":  20,
        "secondary":  [(132, 100, 65), (105, 76, 44), (125, 94, 60), (110, 82, 50)],
        "moisture_shift": (12, 10, 7),
        "texture":    "mixed_grain",
        "has_cracks": False,
        "crack_color": (92, 68, 40),
        "pebble_chance": 0.20,
        "pebble_color": (95, 72, 45),
        "root_chance":  0.30,
    },
    "peat_soil": {
        "base":       (42, 30, 18),
        "variation":  14,
        "secondary":  [(55, 40, 25), (32, 22, 12), (48, 35, 20), (38, 26, 15)],
        "moisture_shift": (5, 8, 4),
        "texture":    "fibrous_organic",
        "has_cracks": False,
        "crack_color": (30, 20, 10),
        "pebble_chance": 0.05,
        "pebble_color": (50, 38, 22),
        "root_chance":  0.50,
    },
    "chalky_soil": {
        "base":       (218, 210, 195),
        "variation":  20,
        "secondary":  [(228, 222, 210), (205, 196, 180), (222, 215, 200), (212, 204, 188)],
        "moisture_shift": (8, 8, 6),
        "texture":    "powdery",
        "has_cracks": True,
        "crack_color": (188, 180, 165),
        "pebble_chance": 0.40,
        "pebble_color": (200, 195, 182),
        "root_chance":  0.08,
    },
    "silt_soil": {
        "base":       (168, 148, 118),
        "variation":  24,
        "secondary":  [(182, 162, 130), (152, 132, 105), (175, 155, 125), (160, 140, 112)],
        "moisture_shift": (14, 12, 10),
        "texture":    "fine_smooth",
        "has_cracks": False,
        "crack_color": (140, 120, 92),
        "pebble_chance": 0.15,
        "pebble_color": (145, 125, 98),
        "root_chance":  0.22,
    },
}


# ── Noise & Texture Utilities ─────────────────────────────────────────────────

def smoothstep(t):
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + t * (b - a)

def generate_noise_grid(rng, size=32):
    return rng.random((size, size)).astype(np.float32)

def upsample_noise(noise_grid, target_size):
    h, w = target_size
    gh, gw = noise_grid.shape
    xs = np.linspace(0, gw - 1, w)
    ys = np.linspace(0, gh - 1, h)
    xi = np.floor(xs).astype(int)
    yi = np.floor(ys).astype(int)
    xi1 = np.clip(xi + 1, 0, gw - 1)
    yi1 = np.clip(yi + 1, 0, gh - 1)
    fx = xs - np.floor(xs)
    fy = ys - np.floor(ys)
    fx = smoothstep(fx)
    fy = smoothstep(fy)
    top    = lerp(noise_grid[np.ix_(yi,  xi )], noise_grid[np.ix_(yi,  xi1)], fx)
    bottom = lerp(noise_grid[np.ix_(yi1, xi )], noise_grid[np.ix_(yi1, xi1)], fx)
    result = lerp(top, bottom, fy[:, None])
    return result


def fractal_noise(rng, size: Tuple[int,int], octaves: int = 5,
                  persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Multi-octave fractal noise for realistic soil texture."""
    h, w = size
    result    = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 1
    max_val   = 0.0
    for _ in range(octaves):
        grid_size = max(4, min(32, int(8 * frequency)))
        noise     = generate_noise_grid(rng, grid_size)
        layer     = upsample_noise(noise, (h, w))
        result   += layer * amplitude
        max_val  += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return result / max_val


def voronoi_texture(rng, size: Tuple[int,int], n_points: int = 80) -> np.ndarray:
    """Voronoi diagram for granular/cell texture."""
    h, w = size
    pts_x = rng.random(n_points) * w
    pts_y = rng.random(n_points) * h
    ys, xs = np.mgrid[0:h, 0:w]
    min_dist = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(n_points):
        dist = np.sqrt((xs - pts_x[i])**2 + (ys - pts_y[i])**2)
        min_dist = np.minimum(min_dist, dist)
    return (min_dist / min_dist.max()).astype(np.float32)


def crack_pattern(rng, size: Tuple[int,int], n_cracks: int = 12) -> np.ndarray:
    """Generate desiccation crack pattern."""
    h, w  = size
    mask  = np.zeros((h, w), dtype=np.float32)
    img   = Image.fromarray((mask * 255).astype(np.uint8))
    draw  = ImageDraw.Draw(img)
    for _ in range(n_cracks):
        x0 = rng.integers(0, w)
        y0 = rng.integers(0, h)
        length = rng.integers(30, 100)
        angle  = rng.uniform(0, 2 * np.pi)
        points = [(x0, y0)]
        for seg in range(rng.integers(3, 8)):
            angle += rng.uniform(-0.6, 0.6)
            seg_len = rng.integers(8, 25)
            nx = int(points[-1][0] + seg_len * np.cos(angle))
            ny = int(points[-1][1] + seg_len * np.sin(angle))
            points.append((nx, ny))
        width = rng.integers(1, 3)
        draw.line(points, fill=255, width=width)
    result = np.array(img).astype(np.float32) / 255.0
    result = np.array(Image.fromarray((result * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))) / 255.0
    return result.astype(np.float32)


def root_fibers(rng, size: Tuple[int,int], n_fibers: int = 18) -> np.ndarray:
    """Simulate root/organic fiber inclusions."""
    h, w = size
    img  = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for _ in range(n_fibers):
        x0    = rng.integers(0, w)
        y0    = rng.integers(0, h)
        color = int(rng.integers(60, 140))
        angle = rng.uniform(0, 2 * np.pi)
        pts   = [(x0, y0)]
        for _ in range(rng.integers(4, 12)):
            angle += rng.uniform(-1.0, 1.0)
            l  = rng.integers(4, 18)
            nx = int(pts[-1][0] + l * np.cos(angle))
            ny = int(pts[-1][1] + l * np.sin(angle))
            pts.append((nx, ny))
        draw.line(pts, fill=color, width=1)
    return np.array(img).astype(np.float32) / 255.0


def add_pebbles(rng, img_array: np.ndarray, pebble_color: Tuple,
                count: int, size_range=(2, 8)) -> np.ndarray:
    """Add small stones/pebbles to soil image."""
    h, w = img_array.shape[:2]
    result = img_array.copy()
    for _ in range(count):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        r  = rng.integers(*size_range)
        yr_start = max(0, cy - r)
        yr_end   = min(h, cy + r)
        xr_start = max(0, cx - r)
        xr_end   = min(w, cx + r)
        for y in range(yr_start, yr_end):
            for x in range(xr_start, xr_end):
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    shade = rng.uniform(0.75, 1.15)
                    result[y, x] = np.clip(
                        np.array(pebble_color) * shade, 0, 255
                    ).astype(np.uint8)
    return result


# ── Core Image Generator ──────────────────────────────────────────────────────

def generate_soil_image(soil_name: str, profile: dict, seed: int,
                         size: Tuple[int,int] = IMG_SIZE) -> Image.Image:
    """Generate one realistic synthetic soil image."""
    rng  = np.random.default_rng(seed)
    h, w = size

    # ── 1. Base color with variation ─────────────────────────────────────
    base = np.array(profile["base"], dtype=np.float32)
    var  = profile["variation"]

    # Choose a secondary color to blend toward
    sec_color = np.array(rng.choice(profile["secondary"]), dtype=np.float32)
    blend     = rng.uniform(0.0, 0.45)
    base_blended = lerp(base, sec_color, blend)

    # ── 2. Moisture level (affects color darkening) ──────────────────────
    moisture = rng.uniform(0.0, 1.0)
    m_shift  = np.array(profile["moisture_shift"], dtype=np.float32) * moisture
    base_col = np.clip(base_blended - m_shift, 0, 255)

    # ── 3. Fractal noise for base texture ────────────────────────────────
    texture_style = profile["texture"]
    if texture_style in ("fine_clay", "smooth_clay", "fine_smooth"):
        octaves, persist = 5, 0.45
    elif texture_style in ("coarse_grain", "rough_granular"):
        octaves, persist = 3, 0.65
    elif texture_style == "fibrous_organic":
        octaves, persist = 6, 0.5
    elif texture_style == "powdery":
        octaves, persist = 4, 0.40
    else:
        octaves, persist = 4, 0.55

    noise = fractal_noise(rng, (h, w), octaves=octaves, persistence=persist)

    # ── 4. Build RGB channels with noise modulation ───────────────────────
    noise_strength = rng.uniform(18, 45)
    img_array = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        channel_noise = fractal_noise(rng, (h, w), octaves=3, persistence=0.5)
        shift = (noise - 0.5) * noise_strength + (channel_noise - 0.5) * 8
        img_array[:, :, c] = base_col[c] + shift

    # ── 5. Voronoi granular overlay ────────────────────────────────────────
    if texture_style in ("granular", "coarse_grain", "rough_granular", "mixed_grain"):
        n_pts = rng.integers(40, 120)
        vor   = voronoi_texture(rng, (h, w), n_pts)
        vor_strength = rng.uniform(8, 22)
        img_array += (vor[:, :, None] - 0.5) * vor_strength

    # ── 6. Layered striations for alluvial/silt ───────────────────────────
    if texture_style == "layered":
        freq = rng.uniform(0.03, 0.08)
        ys   = np.arange(h)
        wave = np.sin(ys * freq * np.pi + rng.uniform(0, np.pi))[:, None]
        img_array += wave * rng.uniform(5, 15)

    # ── 7. Crack patterns ─────────────────────────────────────────────────
    if profile["has_cracks"] and rng.random() > 0.35:
        n_cracks = rng.integers(4, 18)
        cracks   = crack_pattern(rng, (h, w), n_cracks)
        c_col    = np.array(profile["crack_color"], dtype=np.float32)
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * (1 - cracks * 0.7)
                + c_col[c] * (cracks * 0.7)
            )

    # ── 8. Root/organic fibers ─────────────────────────────────────────────
    if rng.random() < profile["root_chance"]:
        n_fibers = rng.integers(5, 25)
        fibers   = root_fibers(rng, (h, w), n_fibers)
        fiber_brightness = rng.uniform(0.55, 0.9)
        for c in range(3):
            img_array[:, :, c] += fibers * fiber_brightness * 45 - fibers * 20

    # ── 9. Clip and convert to uint8 for pebble stage ──────────────────────
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # ── 10. Pebbles / stones ──────────────────────────────────────────────
    if rng.random() < profile["pebble_chance"] * 2:
        n_pebbles = rng.integers(3, 30)
        img_array = add_pebbles(rng, img_array, profile["pebble_color"],
                                n_pebbles, size_range=(2, 9))

    # ── 11. PIL post-processing ───────────────────────────────────────────
    img = Image.fromarray(img_array)

    # Blur to soften pixel noise (simulate camera focus depth)
    blur_radius = rng.uniform(0.3, 1.2)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Brightness / contrast variation
    brightness = rng.uniform(0.78, 1.25)
    contrast   = rng.uniform(0.80, 1.30)
    saturation = rng.uniform(0.70, 1.35)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)

    # Very occasional sharpening (like camera focus)
    if rng.random() > 0.6:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))

    # Slight rotation (field photo angle variation)
    angle = rng.uniform(-4, 4)
    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Final resize ensure exact dimensions
    img = img.resize(size, Image.LANCZOS)

    return img


# ── Batch Generation ──────────────────────────────────────────────────────────

def generate_all(output_dir: str = OUTPUT_DIR,
                 images_per_class: int = IMAGES_PER_CLASS,
                 img_size: Tuple[int,int] = IMG_SIZE):

    total_images  = len(SOIL_PROFILES) * images_per_class
    generated     = 0
    t0            = time.time()

    print("=" * 65)
    print("  IntelliCrop – Soil Image Dataset Generator")
    print(f"  {len(SOIL_PROFILES)} classes × {images_per_class} images = {total_images} total")
    print(f"  Image size: {img_size[0]}×{img_size[1]} px  |  Output: {output_dir}")
    print("=" * 65)

    summary = {}

    for soil_idx, (soil_name, profile) in enumerate(SOIL_PROFILES.items()):
        class_dir = os.path.join(output_dir, soil_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n[{soil_idx+1:02d}/{len(SOIL_PROFILES)}] {soil_name.replace('_',' ').title()}")
        print(f"  Output: {class_dir}")

        class_count = 0
        for i in range(images_per_class):
            seed = SEED_BASE + soil_idx * 100000 + i
            img  = generate_soil_image(soil_name, profile, seed, img_size)

            filename  = f"{soil_name}_{i+1:04d}.jpg"
            filepath  = os.path.join(class_dir, filename)
            img.save(filepath, "JPEG", quality=92, optimize=True)

            class_count += 1
            generated   += 1

            # Progress bar
            if (i + 1) % 50 == 0 or (i + 1) == images_per_class:
                bar_len = 30
                filled  = int(bar_len * (i + 1) / images_per_class)
                bar     = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - t0
                rate    = generated / elapsed if elapsed > 0 else 0
                eta     = (total_images - generated) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  [{bar}] {i+1}/{images_per_class} "
                    f"| {rate:.1f} img/s | ETA {eta:.0f}s  "
                )
                sys.stdout.flush()

        summary[soil_name] = class_count
        print(f"\n  ✅ {class_count} images saved")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Total images   : {generated:,}")
    print(f"  Time elapsed   : {elapsed:.1f}s  ({generated/elapsed:.1f} img/s)")
    print(f"  Output dir     : {output_dir}")
    print()
    print(f"  {'Class':<25} {'Count':>6}  {'Size on Disk':>12}")
    print(f"  {'-'*25} {'-'*6}  {'-'*12}")
    total_size = 0
    for soil_name, count in summary.items():
        class_dir  = os.path.join(output_dir, soil_name)
        dir_size   = sum(
            os.path.getsize(os.path.join(class_dir, f))
            for f in os.listdir(class_dir)
            if f.endswith(".jpg")
        )
        total_size += dir_size
        label = soil_name.replace("_", " ").title()
        print(f"  {label:<25} {count:>6}  {dir_size/1024/1024:>10.1f} MB")
    print(f"  {'─'*25} {'─'*6}  {'─'*12}")
    print(f"  {'TOTAL':<25} {generated:>6}  {total_size/1024/1024:>10.1f} MB")
    print()

    return summary


if __name__ == "__main__":
    generate_all(
        output_dir      = OUTPUT_DIR,
        images_per_class = IMAGES_PER_CLASS,
        img_size        = IMG_SIZE,
    )

"""Connector end-face inspection images and OTDR trace plots.

End faces follow the usual inspection-scope view: bright cladding disc with
the core at the centre, adhesive ring and ferrule around it. Dirty
connectors carry particles and film; damaged ones carry scratches through
the core zone and chips at the cladding edge. A small numerical analyser is
included so the reference solver can grade images without a vision model.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SIZE = 320
C = SIZE // 2
R_CORE, R_CLAD, R_RING = 7, 105, 118


def render_endface(state: str, seed: int, path: str) -> str:
    rng = np.random.default_rng([seed, 777, {"clean": 0, "dirty": 1, "damaged": 2}[state]])
    img = Image.new("L", (SIZE, SIZE), 40)
    d = ImageDraw.Draw(img)
    d.ellipse([C - 150, C - 150, C + 150, C + 150], fill=95)           # ferrule
    d.ellipse([C - R_RING, C - R_RING, C + R_RING, C + R_RING], fill=70)  # adhesive ring
    d.ellipse([C - R_CLAD, C - R_CLAD, C + R_CLAD, C + R_CLAD], fill=175)  # cladding
    d.ellipse([C - R_CORE, C - R_CORE, C + R_CORE, C + R_CORE], fill=215)  # core

    if state == "clean":
        # A harmless speck on the ferrule, outside the graded zones.
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(125, 140)
        x, y = C + r * np.cos(a), C + r * np.sin(a)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=55)
    elif state == "dirty":
        for _ in range(int(rng.integers(6, 14))):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 95) ** 1.0
            x, y = C + r * np.cos(a), C + r * np.sin(a)
            rx, ry = rng.uniform(3, 11), rng.uniform(3, 11)
            ry = min(max(ry, rx / 1.4), rx * 1.4)
            d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=int(rng.integers(35, 80)))
        # Film smudge across part of the cladding.
        film = Image.new("L", (SIZE, SIZE), 0)
        fd = ImageDraw.Draw(film)
        fx, fy = C + rng.uniform(-40, 40), C + rng.uniform(-40, 40)
        fd.ellipse([fx - 45, fy - 30, fx + 45, fy + 30], fill=45)
        film = film.filter(ImageFilter.GaussianBlur(10))
        img = Image.fromarray(np.clip(np.asarray(img, float) - np.asarray(film, float), 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
    elif state == "damaged":
        for i in range(int(rng.integers(1, 4))):
            a = rng.uniform(0, np.pi)
            off = rng.uniform(-12, 12) if i == 0 else rng.uniform(-60, 60)
            length = rng.uniform(110, 190)
            nx, ny = -np.sin(a), np.cos(a)
            cx, cy = C + off * nx, C + off * ny
            x0, y0 = cx - length / 2 * np.cos(a), cy - length / 2 * np.sin(a)
            x1, y1 = cx + length / 2 * np.cos(a), cy + length / 2 * np.sin(a)
            d.line([x0, y0, x1, y1], fill=int(rng.integers(25, 60)), width=int(rng.integers(2, 4)))
        # Chip at the cladding edge.
        a = rng.uniform(0, 2 * np.pi)
        pts = []
        for da, rr in ((-0.18, R_CLAD + 2), (0.0, R_CLAD - rng.uniform(18, 30)), (0.18, R_CLAD + 2)):
            pts.append((C + rr * np.cos(a + da), C + rr * np.sin(a + da)))
        d.polygon(pts, fill=45)

    arr = np.asarray(img, float)
    arr += np.random.default_rng([seed, 778]).normal(0, 4.0, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def analyse_endface(path: str) -> dict:
    """Simple numerical inspection: find dark defects inside the cladding and
    describe them by size and elongation."""
    arr = np.asarray(Image.open(path).convert("L"), float)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rr = np.hypot(xx - C, yy - C)
    zone = (rr < R_CLAD - 3) & (rr > R_CORE + 2)
    defects = zone & (arr < 135)
    labels, n = ndimage.label(defects)
    blobs, lines = 0, 0
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if ys.size < 12:
            continue
        cov = np.cov(np.vstack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong = np.sqrt(ev[1] / max(ev[0], 1e-6))
        extent = 4.0 * np.sqrt(ev[1])
        thinness = ys.size / max(extent, 1.0) ** 2
        if extent > 40 and (elong > 4.0 or thinness < 0.12):
            lines += 1
        else:
            blobs += 1
    if lines:
        verdict = "damaged"
    elif blobs:
        verdict = "dirty"
    else:
        verdict = "clean"
    return {"defect_regions": int(blobs + lines), "linear_defects": int(lines), "verdict": verdict}


def render_trace(trace: dict, title: str, path: str, start_km=None, end_km=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, y = trace["distance_km"], trace["level_db"]
    m = np.ones_like(z, bool)
    if start_km is not None:
        m &= z >= start_km
    if end_km is not None:
        m &= z <= end_km
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    ax.plot(z[m], y[m], lw=0.8, color="#1f4e79")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Level (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

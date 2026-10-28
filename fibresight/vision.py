"""Connector end-face defect detection with SAM 3.1 (Meta Model API).

SAM 3.1 takes an image and a short text concept and returns a box and a
mask for every match. We ask for two concepts, "dark spot" (particles,
contamination) and "dark line" (scratches, cracks), then grade each
detection by where it sits, following the zone idea of IEC 61300-3-35:
defects inside the cladding disc matter, specks on the ferrule do not.
"""
from __future__ import annotations

import base64
import math
import os
import re

from PIL import Image, ImageDraw

from .images import C, R_CLAD, R_CORE

BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")
CONCEPTS = {"particle": "dark spot", "scratch": "dark line"}


def _sam_boxes(client, path: str, phrase: str) -> list:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    r = client.responses.create(model="sam-3.1", input=[{"role": "user", "content": [
        {"type": "input_text", "text": phrase},
        {"type": "input_image", "image_url": "data:image/png;base64," + b64}]}])
    out = []
    for m in BOX.findall(r.output_text or ""):
        x1, y1, x2, y2, w, h = map(int, m)
        sx, sy = Image.open(path).size[0] / w, Image.open(path).size[1] / h
        out.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out


def _zone(box) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    r = math.hypot(cx - C, cy - C)
    # A long scratch can pass through the core even if its centre does not.
    reach = min(math.hypot(x - C, y - C) for x in (box[0], box[2], cx) for y in (box[1], box[3], cy))
    if r <= R_CORE * 3 or reach <= R_CORE * 2:
        return "core"
    if r <= R_CLAD:
        return "cladding"
    return "ferrule"


def segment_endface(client, path: str, out_path: str) -> dict:
    """Run SAM 3.1 on an end-face image, grade it and save an annotated copy."""
    found = []
    for kind, phrase in CONCEPTS.items():
        for b in _sam_boxes(client, path, phrase):
            found.append({"kind": kind, "box": [round(v) for v in b], "zone": _zone(b)})
    graded = [f for f in found if f["zone"] in ("core", "cladding")]
    scratches = [f for f in graded if f["kind"] == "scratch"]
    particles = [f for f in graded if f["kind"] == "particle"]
    if scratches:
        verdict = "FAIL: scratch or crack in the core/cladding zone (damaged, replace)"
    elif particles:
        verdict = "FAIL: contamination in the core/cladding zone (clean and re-inspect)"
    else:
        verdict = "PASS: no defects in the core/cladding zone"

    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([C - R_CLAD, C - R_CLAD, C + R_CLAD, C + R_CLAD], outline=(80, 160, 255), width=1)
    for f in found:
        colour = (255, 60, 60) if f["kind"] == "scratch" else (255, 200, 0)
        if f["zone"] == "ferrule":
            colour = (140, 140, 140)
        d.rectangle(f["box"], outline=colour, width=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    return {"model": "sam-3.1", "concepts": list(CONCEPTS.values()),
            "particles_in_zone": len(particles), "scratches_in_zone": len(scratches),
            "outside_zone": len(found) - len(graded), "detections": found,
            "verdict": verdict, "annotated_image_path": out_path}

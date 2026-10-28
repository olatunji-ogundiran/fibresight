"""OTDR model: backscatter trace synthesis and an event table that behaves
like real OTDR analysis software (detection threshold, merged events in the
pulse dead zone, direction-dependent splice losses and ghost reflections).

Distances are fibre distances from the ODF at the shooting end. Launch and
receive cords (1 km each) are assumed, so both ODF connectors appear as
events; the trace ends at the open end of the receive cord.
"""
from __future__ import annotations

import numpy as np

from .sim import NIGHT_TEST, NOW, Case

RECEIVE_CORD_KM = 1.0
OPEN_END_REFLECTANCE = -14.5
BACKSCATTER_1NS = {1310: -79.5, 1550: -81.5, 1625: -82.0}


def otdr_settings(case: Case) -> dict:
    if case.fibre_length_km <= 25.0:
        return {"pulse_ns": 100, "sample_km": 0.005, "dynamic_range_db": {1310: 26.0, 1550: 28.0, 1625: 27.0}}
    return {"pulse_ns": 1000, "sample_km": 0.02, "dynamic_range_db": {1310: 36.0, 1550: 38.0, 1625: 37.0}}


def _pulse_km(pulse_ns: float) -> float:
    return pulse_ns * 1e-9 * 2.0e8 / 2.0 / 1000.0  # spatial pulse length (one way)


def test_time(when: str):
    return NIGHT_TEST if when == "night" else NOW


def physical_events(case: Case, wl: int, from_site: str, when: str = "now") -> tuple:
    """Events seen from one end, nearest first, plus the end position.

    Returns (events, end_km, end_reflectance). Each event is a dict with
    pos (km from the shooting ODF), loss (apparent, this direction),
    refl (dB or None) and src (hidden source id, for debugging only).
    """
    F = case.fibre_length_km
    evs = []
    for e in case.elements:
        pos_a = e.fibre_km
        if e.kind == "connector":
            evs.append({"pos_a": pos_a, "loss": e.loss[wl], "refl": e.reflectance, "src": e.id})
        else:
            apparent = e.loss[wl] + (e.mfd_offset[wl] if from_site == "A" else -e.mfd_offset[wl])
            evs.append({"pos_a": pos_a, "loss": apparent, "refl": None, "src": e.id})
    f = case.bend_factor(test_time(when))
    for ev in case.events:
        scale = f if ev["kind"] == "bend" else 1.0
        evs.append({"pos_a": ev["fibre_km"], "loss": ev["loss"][wl] * scale, "refl": None, "src": ev["kind"]})

    if case.break_fibre_km is not None:
        b = case.break_fibre_km
        if from_site == "A":
            evs = [x for x in evs if x["pos_a"] < b]
            end, end_refl = b, case.break_reflectance
        else:
            evs = [x for x in evs if x["pos_a"] > b]
            end, end_refl = F - b, case.break_reflectance
    else:
        end, end_refl = F + RECEIVE_CORD_KM, OPEN_END_REFLECTANCE

    out = []
    for x in evs:
        pos = x["pos_a"] if from_site == "A" else F - x["pos_a"]
        out.append({"pos": pos, "loss": x["loss"], "refl": x["refl"], "src": x["src"]})
    out.sort(key=lambda x: x["pos"])
    return out, end, end_refl


def _ghosts(events: list, end: float) -> list:
    ghosts = []
    for ev in events:
        if ev["refl"] is not None and ev["refl"] > -40.0 and ev["pos"] > 0.05:
            g = 2.0 * ev["pos"]
            if g < end - 0.05:
                ghosts.append({"pos": g, "loss": 0.0, "refl": ev["refl"] - 18.0, "src": "ghost"})
    return ghosts


def synthesize(case: Case, wl: int, from_site: str, seed_offset: int = 0, when: str = "now") -> dict:
    """Backscatter trace in dB (5*log10 display convention) vs distance."""
    st = otdr_settings(case)
    rng = np.random.default_rng([case.seed, wl, 1 if from_site == "A" else 2, seed_offset, 1 if when == "night" else 0])
    events, end, end_refl = physical_events(case, wl, from_site, when)
    ghosts = _ghosts(events, end)
    dz = st["sample_km"]
    pulse = _pulse_km(st["pulse_ns"])
    z = np.arange(-0.2, end + 1.5, dz)
    alpha = case.alpha[wl]

    y = -alpha * np.clip(z, 0.0, None)
    for ev in events:
        ramp = np.clip((z - ev["pos"]) / pulse, 0.0, 1.0)
        y -= ev["loss"] * ramp
    p_back = 10.0 ** (y / 5.0)
    p_back[z < 0.0] = 10.0 ** (-0.2)
    p_back[z > end] = 0.0

    b = BACKSCATTER_1NS[wl] + 10.0 * np.log10(st["pulse_ns"])
    p = p_back.copy()
    shape = lambda pos: np.exp(-np.clip(z - pos, 0, None) / (0.35 * pulse)) * (z >= pos - 0.25 * pulse)
    for ev in events + ghosts:
        if ev["refl"] is None:
            continue
        idx = min(max(int(np.searchsorted(z, ev["pos"])), 0), len(z) - 1)
        p += p_back[max(idx - 1, 0)] * 10.0 ** ((ev["refl"] - b) / 10.0) * shape(ev["pos"])
    if end_refl is not None:
        idx = min(max(int(np.searchsorted(z, end)) - 1, 0), len(z) - 1)
        p += p_back[idx] * 10.0 ** ((end_refl - b) / 10.0) * shape(end)

    floor = 10.0 ** (-st["dynamic_range_db"][wl] / 5.0)
    noise = floor * np.abs(1.0 + 0.6 * rng.standard_normal(z.size))
    p_meas = np.abs(p * (1.0 + 0.002 * rng.standard_normal(z.size)) + noise)
    level = 5.0 * np.log10(p_meas)
    level = np.minimum(level, 1.5)  # receiver saturation clips big reflections
    return {"distance_km": z, "level_db": level, "settings": st, "end_km": end}


def event_table(case: Case, wl: int, from_site: str, when: str = "now") -> dict:
    """What the OTDR's automatic analysis reports for this measurement."""
    st = otdr_settings(case)
    rng = np.random.default_rng([case.seed, wl, 11 if from_site == "A" else 12, 1 if when == "night" else 0])
    events, end, end_refl = physical_events(case, wl, from_site, when)
    ghosts = _ghosts(events, end)
    pulse = _pulse_km(st["pulse_ns"])
    dr = st["dynamic_range_db"][wl]
    alpha = case.alpha[wl]

    # Merge events closer than the event dead zone.
    merged = []
    for ev in sorted(events + ghosts, key=lambda x: x["pos"]):
        if merged and ev["pos"] - merged[-1]["pos"] < 1.5 * pulse and ev["src"] != "ghost" and merged[-1]["src"] != "ghost":
            m = merged[-1]
            m["loss"] += ev["loss"]
            if ev["refl"] is not None:
                m["refl"] = ev["refl"] if m["refl"] is None else max(m["refl"], ev["refl"])
            m["src"] += "+" + ev["src"]
        else:
            merged.append(dict(ev))

    rows, depth = [], 0.0
    for ev in merged:
        depth_here = alpha * ev["pos"] + depth
        snr_penalty = max(0.0, (depth_here - (dr - 12.0)) / 4.0)
        loss_meas = ev["loss"] + rng.normal(0.0, 0.015 + 0.03 * snr_penalty)
        pos_meas = ev["pos"] + rng.normal(0.0, st["sample_km"] / 3.0)
        reflective = ev["refl"] is not None
        if not reflective and abs(loss_meas) < 0.04:
            depth += ev["loss"]
            continue
        rows.append({
            "distance_km": round(float(pos_meas), 3),
            "type": "reflective" if reflective else "non-reflective",
            "loss_db": round(float(loss_meas), 2),
            "reflectance_db": round(float(ev["refl"] + rng.normal(0.0, 0.8)), 1) if reflective else None,
        })
        depth += ev["loss"]

    end_row = {"distance_km": round(float(end + rng.normal(0.0, st["sample_km"] / 3.0)), 3),
               "type": "end",
               "loss_db": None,
               "reflectance_db": round(float(end_refl + rng.normal(0, 0.8)), 1) if end_refl is not None else None}
    rows.append(end_row)
    for i, r in enumerate(rows, 1):
        r["event"] = i
    return {
        "wavelength_nm": wl,
        "from_site": from_site,
        "test_time": test_time(when).strftime("%Y-%m-%d %H:%M UTC"),
        "pulse_width_ns": st["pulse_ns"],
        "group_index": 1.4682,
        "note": "Distances are fibre distances from the ODF at the shooting site. Launch and receive cords of 1.000 km were used; the receive cord ends in an open connector.",
        "fibre_attenuation_db_per_km": round(float(alpha + rng.normal(0, 0.003)), 3),
        "events": [{k: r[k] for k in ("event", "distance_km", "type", "loss_db", "reflectance_db")} for r in rows],
    }

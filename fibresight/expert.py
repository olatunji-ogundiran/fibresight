"""Reference expert solver. It uses only the public tools, the way a senior
transmission engineer would:

1. Read alarms and the record. Total loss of signal points to a break.
2. Check the equipment first with a power meter: transmitter output and
   the attenuator (power before and after it). Rising laser bias with
   falling output means the transmitter is at end of life.
3. Otherwise shoot the OTDR from both ends at 1310 and 1550 nm, average
   each event's loss over both directions (cancels splice mode-field
   mismatch), drop one-direction ghosts, and take the lossiest event.
4. Classify it: reflective events are connectors (inspect the end face:
   dirty vs damaged); non-reflective events with much more loss at 1550
   than 1310 are macrobends, otherwise splices.
5. Convert OTDR fibre distance to route distance by anchoring on the splice
   closures seen in the trace (slack loops and cable helix make fibre longer
   than route), then report the element or the route km.
"""
from __future__ import annotations

from . import images
from .sim import TX_SPEC
from .tools import ToolBox

NOMINAL_SLACK_KM = 0.050
NOMINAL_EXCESS = 0.014
MATCH_TOL_KM = 0.12


def _event_positions(tables: list) -> list:
    """Union of event positions (non-end) across tables in one frame, clustered."""
    pos = sorted(ev["distance_km"] for t in tables for ev in t["events"] if ev["type"] != "end")
    clusters = []
    for p in pos:
        if clusters and p - clusters[-1][-1] < 0.03:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [sum(c) / len(c) for c in clusters]


def _anchors(record: dict, frame: str, positions: list, fibre_len: float = None) -> list:
    """Match route elements to OTDR event positions walking away from the
    shooting end. Returns anchors [(route_km_from_A, fibre_km_in_frame, id, slack_km, matched)]."""
    elems = record["elements_in_route_order"]
    L = elems[-1]["route_km"]
    seq = elems if frame == "A" else list(reversed(elems))
    anchors = []
    prev_r, prev_f, prev_slack = None, None, 0.0
    for e in seq:
        r_from = e["route_km"] if frame == "A" else L - e["route_km"]
        slack = NOMINAL_SLACK_KM if e["type"].startswith("splice") else 0.0
        if prev_r is None:
            anchors.append((e["route_km"], 0.0, e["id"], slack, True))
            prev_r, prev_f, prev_slack = r_from, 0.0, slack
            continue
        pred = prev_f + prev_slack / 2 + (r_from - prev_r) * (1 + NOMINAL_EXCESS) + slack / 2
        if fibre_len is not None and e is seq[-1]:
            f, matched = fibre_len, True
        else:
            near = [p for p in positions if abs(p - pred) < MATCH_TOL_KM]
            if near:
                f, matched = min(near, key=lambda p: abs(p - pred)), True
            else:
                f, matched = pred, False
        anchors.append((e["route_km"], f, e["id"], slack, matched))
        prev_r, prev_f, prev_slack = r_from, f, slack
    return anchors


def _fibre_to_route(anchors: list, f: float, frame: str, L: float) -> float:
    """Interpolate a fibre position (frame coordinates) to route km from A."""
    for (r0, f0, _, s0, _), (r1, f1, _, s1, _) in zip(anchors[:-1], anchors[1:]):
        if f0 <= f <= f1:
            a, b = f0 + s0 / 2, f1 - s1 / 2
            frac = min(max((f - a) / max(b - a, 1e-6), 0.0), 1.0)
            return r0 + frac * (r1 - r0)
    # Beyond the last anchor: extrapolate with nominal excess length.
    r0, f0, _, s0, _ = anchors[-1]
    d = (f - f0 - s0 / 2) / (1 + NOMINAL_EXCESS)
    return r0 + d if frame == "A" else r0 - d


def _element_at(anchors: list, f: float, tol: float = 0.04):
    for r, fa, eid, _, matched in anchors:
        if matched and abs(fa - f) <= tol:
            return eid, r
    return None, None


def solve(tb: ToolBox, equipment: bool = True, bidirectional: bool = True, dual_wavelength: bool = True,
          anchors_on: bool = True, inspect: bool = True) -> dict:
    """Flags switch off single expert steps (used for ablation baselines)."""
    record = tb.call("get_link_record", {})
    L = record["elements_in_route_order"][-1]["route_km"]
    alarms = tb.call("get_alarms", {})["alarms"]
    active = {a["alarm"] for a in alarms if a["cleared"] == "active"}
    # Alarms that keep raising and clearing (mostly at night) point to an
    # intermittent, temperature-driven loss: test in tonight's window.
    low = [a for a in alarms if a["alarm"].startswith("RX_LOW")]
    when = "night" if len(low) >= 3 and sum(a["cleared"] != "active" for a in low) >= 2 else "now"
    hard = tb.mode == "hard"

    light_at_b = (tb.call("measure_power", {"point": "B-ODF"})["power_dbm"] is not None) if equipment else False
    if "RX_LOS" in active and not light_at_b:
        if not anchors_on:
            end_a = tb.call("otdr_event_table", {"wavelength_nm": 1550, "from_site": "A"})["events"][-1]["distance_km"]
            return {"fault_type": "fibre_break", "element_id": None, "route_km": round(end_a, 3),
                    "action": "repair_break", "evidence": "OTDR end distance used as route km."}
        ta = [tb.call("otdr_event_table", {"wavelength_nm": wl, "from_site": "A"}) for wl in (1550, 1310)]
        tbb = [tb.call("otdr_event_table", {"wavelength_nm": wl, "from_site": "B"}) for wl in ((1550,) if hard else (1550, 1310))]
        end_a = ta[0]["events"][-1]["distance_km"]
        end_b = tbb[0]["events"][-1]["distance_km"]
        anc_a = _anchors(record, "A", _event_positions(ta))
        anc_a = [a for a in anc_a if a[1] < end_a - 0.02]
        anc_b = _anchors(record, "B", _event_positions(tbb))
        anc_b = [a for a in anc_b if a[1] < end_b - 0.02]
        r_a = _fibre_to_route(anc_a, end_a, "A", L)
        r_b = _fibre_to_route(anc_b, end_b, "B", L)
        r = 0.5 * (r_a + r_b)
        return {"fault_type": "fibre_break", "element_id": None, "route_km": round(r, 3),
                "action": "repair_break",
                "evidence": f"RX_LOS active; OTDR ends at {end_a:.3f} km fibre from A and {end_b:.3f} km from B; mapped to route km via closure anchors."}

    if not equipment:
        return _link_diagnosis(tb, record, L, bidirectional, dual_wavelength, anchors_on, inspect, when)
    tx = tb.call("measure_power", {"point": "A-TX"})["power_dbm"]
    bias = tb.call("get_pm", {"metric": "tx_bias_ma", "resolution": "daily"})["rows"]
    bias_rise = bias[-1]["max"] - bias[0]["avg"]
    txh = tb.call("get_pm", {"metric": "tx_power_dbm", "resolution": "daily"})["rows"]
    tx_drop = txh[0]["avg"] - txh[-1]["min"]
    if (tx < TX_SPEC[0] - 0.3 or tx_drop > 1.5) and bias_rise > 10.0:
        return {"fault_type": "tx_degradation", "element_id": "A-TX", "route_km": None,
                "action": "replace_transmitter",
                "evidence": f"Tx output {tx:.2f} dBm below spec; laser bias rose {bias_rise:.1f} mA over 14 days."}

    p_odf = tb.call("measure_power", {"point": "B-ODF"})["power_dbm"]
    p_rx = tb.call("measure_power", {"point": "B-RX"})["power_dbm"]
    att_meas = p_odf - p_rx - 0.1
    design = float(record["service"]["B-ATT"].split("design value")[1].split("dB")[0])
    if abs(att_meas - design) >= 2.5:
        return {"fault_type": "wrong_attenuator", "element_id": "B-ATT", "route_km": None,
                "action": "replace_attenuator",
                "evidence": f"Attenuator measures {att_meas:.1f} dB against a design value of {design:.0f} dB."}
    return _link_diagnosis(tb, record, L, bidirectional, dual_wavelength, anchors_on, inspect, when)


def _link_diagnosis(tb, record, L, bidirectional, dual_wavelength, anchors_on, inspect, when="now") -> dict:
    sites = ("A", "B") if bidirectional else ("A",)
    wls = (1310, 1550) if dual_wavelength else (1550,)
    plan = [(wl, s) for wl in wls for s in sites]
    if tb.mode == "hard":  # three tests: 1550 both ends, 1310 from A
        plan = [p for p in plan if not (p[0] == 1310 and p[1] == "B")]
    tabs = {(wl, s): tb.call("otdr_event_table", {"wavelength_nm": wl, "from_site": s, "time": when})
            for wl, s in plan}
    F = tabs[(1550, "A")]["events"][-1]["distance_km"] - 1.0
    # Bidirectional event list in the A frame.
    def evs(wl, s):
        out = []
        if (wl, s) not in tabs:
            return out
        for ev in tabs[(wl, s)]["events"]:
            if ev["type"] == "end":
                continue
            pos = ev["distance_km"] if s == "A" else F - ev["distance_km"]
            out.append(dict(ev, pos=pos))
        return out

    def match(lst, pos, tol=0.06):
        c = [e for e in lst if abs(e["pos"] - pos) < tol]
        return min(c, key=lambda e: abs(e["pos"] - pos)) if c else None

    cands = []
    for ev in evs(1550, "A") + evs(1550, "B"):
        if any(abs(c["pos"] - ev["pos"]) < 0.06 for c in cands):
            continue
        a15, b15 = match(evs(1550, "A"), ev["pos"]), match(evs(1550, "B"), ev["pos"])
        a13, b13 = match(evs(1310, "A"), ev["pos"]), match(evs(1310, "B"), ev["pos"])
        if not bidirectional:
            if a15 is None:
                continue
            l15, l13 = a15["loss_db"], (a13["loss_db"] if a13 else None)
            refl = a15["type"] == "reflective"
            cands.append({"pos": ev["pos"], "l1550": l15, "l1310": l13, "reflective": refl})
            continue
        if a15 is None or b15 is None:
            continue  # seen from one end only: ghost or sub-threshold
        l15 = (a15["loss_db"] + b15["loss_db"]) / 2
        if not dual_wavelength:
            l13 = None
        elif (1310, "B") in tabs:
            l13 = ((a13["loss_db"] if a13 else 0.0) + (b13["loss_db"] if b13 else 0.0)) / 2
        else:
            l13 = a13["loss_db"] if a13 else 0.0
        refl = a15["type"] == "reflective" or b15["type"] == "reflective"
        cands.append({"pos": ev["pos"], "l1550": l15, "l1310": l13, "reflective": refl})
    worst = max(cands, key=lambda c: c["l1550"])

    positions = _event_positions([t for (wl, s), t in tabs.items() if s == "A"])
    if bidirectional:
        positions += [F - p for p in _event_positions([t for (wl, s), t in tabs.items() if s == "B"])]
    positions = sorted(positions)
    if anchors_on:
        anchors = _anchors(record, "A", positions, fibre_len=F)
    else:  # naive: treat fibre distance as route distance
        anchors = [(e["route_km"], e["route_km"], e["id"], 0.0, True) for e in record["elements_in_route_order"]]
    eid, r_el = _element_at(anchors, worst["pos"], tol=0.04 if anchors_on else 0.25)

    if worst["reflective"]:
        if eid is None:
            eid = min((a for a in anchors if a[2] in ("A-ODF", "MID-PP", "B-ODF")),
                      key=lambda a: abs(a[1] - worst["pos"]))[2]
        if inspect:
            img = tb.call("inspect_connector", {"element_id": eid})
            verdict = images.analyse_endface(img["image_path"])["verdict"]
        else:
            verdict = "dirty (assumed, not inspected)"
        ft = "damaged_connector" if verdict == "damaged" else "dirty_connector"
        return {"fault_type": ft, "element_id": eid, "route_km": None,
                "action": "replace_connector" if ft == "damaged_connector" else "clean_connector",
                "evidence": f"Reflective event at {worst['pos']:.3f} km with {worst['l1550']:.2f} dB bidirectional loss; end face {verdict}."}
    if worst["l1310"] is None:  # single wavelength: guess bend unless at a closure
        is_bend = eid is None
    else:
        is_bend = worst["l1550"] / max(worst["l1310"], 0.05) > 1.8
    if is_bend:
        r = r_el if r_el is not None else (_fibre_to_route(anchors, worst["pos"], "A", L) if anchors_on else worst["pos"])
        return {"fault_type": "macrobend", "element_id": eid, "route_km": round(r, 3), "action": "relieve_bend",
                "evidence": f"Non-reflective loss {worst['l1550']:.2f} dB at 1550 nm, much lower at 1310 nm."}
    if eid is None:
        eid = min((a for a in anchors if a[2].startswith("SC-")), key=lambda a: abs(a[1] - worst["pos"]))[2]
    return {"fault_type": "bad_splice", "element_id": eid, "route_km": None, "action": "resplice",
            "evidence": f"Non-reflective loss {worst['l1550']:.2f} dB at 1550 nm at closure {eid}."}

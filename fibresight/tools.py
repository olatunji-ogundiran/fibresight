"""The tool interface an agent uses to investigate one case.

Every tool returns JSON-serialisable data. Image tools also return an
``image_path`` that the agent harness attaches to the conversation, so a
multimodal model can look at the OTDR trace or the connector end face.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from . import images, otdr, pm
from .sim import (ATTENUATOR_VALUES, NOW, PM_DAYS, RX_HIGH_ALARM, RX_HIGH_WARN, RX_LOS, RX_LOW_ALARM,
                  RX_LOW_WARN, RX_OVERLOAD, RX_SENSITIVITY, TX_SPEC, Case, link_loss)

CONNECTOR_IDS = ("A-ODF", "MID-PP", "B-ODF")
POWER_POINTS = ("A-TX", "B-ODF", "B-RX")


class ToolError(Exception):
    pass


class ToolBox:
    def __init__(self, case: Case, workdir: str, mode: str = "standard"):
        """mode: 'standard' (OTDR auto event table available) or 'raw' (traces only)."""
        self.case = case
        self.mode = mode
        self.specs = [t for t in TOOL_SPECS if not (mode == "raw" and t["name"] == "otdr_event_table")]
        self.names = {t["name"] for t in self.specs}
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self._hist = pm.history(case)
        self.calls = []

    # ------------------------------------------------------------ tools
    def get_link_record(self) -> dict:
        c = self.case
        elems = []
        for e in c.elements:
            row = {"id": e.id, "type": "connector (ODF/patch panel)" if e.kind == "connector" else "splice closure",
                   "route_km": round(e.route_km, 3)}
            if e.kind == "splice":
                row["slack_stored_m"] = "about 50 (as-built varies)"
            if e.note:
                row["note"] = e.note
            elems.append(row)
        return {
            "link": f"Site A to Site B, single-mode G.652.D, route length {c.route_km:.2f} km (as-built route map)",
            "elements_in_route_order": elems,
            "cable": {"fibre_excess_length_over_cable_pct": "1.0 to 1.8 (manufacturer spec)",
                      "slack_loops": "spare fibre is coiled at each splice closure"},
            "service": {
                "direction": "site A transmits, site B receives, 1550 nm",
                "A-TX": f"10G 1550 nm transceiver, launch power spec {TX_SPEC[0]:+.1f} to {TX_SPEC[1]:+.1f} dBm",
                "B-ATT": f"fixed in-line attenuator at site B between ODF and receiver, design value {c.att_design_db:.0f} dB",
                "B-RX": f"receiver sensitivity {RX_SENSITIVITY:.1f} dBm, overload {RX_OVERLOAD:.1f} dBm",
                "available_attenuator_values_db": list(ATTENUATOR_VALUES),
            },
            "commissioning_link_loss_db": {"1310": c.commissioning_loss[1310], "1550": c.commissioning_loss[1550],
                                           "note": "A-ODF to B-ODF, bidirectional average, at commissioning"},
            "alarm_thresholds_dbm": {"RX_LOW_WARNING": RX_LOW_WARN, "RX_LOW_ALARM": RX_LOW_ALARM, "RX_LOS": RX_LOS,
                                     "RX_HIGH_WARNING": RX_HIGH_WARN, "RX_HIGH_ALARM": RX_HIGH_ALARM},
            "answer_ids": {"elements": [e.id for e in c.elements], "equipment": ["A-TX", "B-ATT"]},
        }

    def get_alarms(self) -> dict:
        return {"now": NOW.strftime("%Y-%m-%d %H:%M UTC"), "alarms": pm.alarms(self.case, self._hist)}

    def get_pm(self, metric: str, resolution: str = "daily", last_hours: Optional[int] = None) -> dict:
        if metric not in pm.METRICS:
            raise ToolError(f"metric must be one of {list(pm.METRICS)}")
        h = self._hist
        vals, times = np.asarray(h[metric]), h["time"]
        fmt = (lambda v: float(f"{v:.3g}")) if metric == "pre_fec_ber" else (lambda v: round(float(v), 2))
        if resolution == "daily":
            rows = []
            for d in range(PM_DAYS):
                seg = vals[d * 24:(d + 1) * 24]
                rows.append({"day": times[d * 24].strftime("%Y-%m-%d"), "min": fmt(seg.min()),
                             "avg": fmt(np.exp(np.log(seg).mean()) if metric == "pre_fec_ber" else seg.mean()),
                             "max": fmt(seg.max())})
            return {"metric": metric, "resolution": "daily (UTC days, 24 hourly samples each)", "rows": rows}
        if resolution == "hourly":
            n = int(min(max(last_hours or 48, 1), 72))
            rows = [{"time": times[k].strftime("%Y-%m-%d %H:%M"), "value": fmt(vals[k])}
                    for k in range(len(vals) - n, len(vals))]
            return {"metric": metric, "resolution": f"hourly, last {n} hours", "rows": rows}
        raise ToolError("resolution must be 'daily' or 'hourly'")

    def otdr_event_table(self, wavelength_nm: int, from_site: str) -> dict:
        wl, site = self._check_otdr(wavelength_nm, from_site)
        return otdr.event_table(self.case, wl, site)

    def otdr_trace(self, wavelength_nm: int, from_site: str, start_km: float = None,
                   end_km: float = None, points: int = 120) -> dict:
        wl, site = self._check_otdr(wavelength_nm, from_site)
        tr = otdr.synthesize(self.case, wl, site)
        z, y = tr["distance_km"], tr["level_db"]
        lo = z[0] if start_km is None else float(start_km)
        hi = z[-1] if end_km is None else float(end_km)
        m = (z >= lo) & (z <= hi)
        if m.sum() < 2:
            raise ToolError("empty distance window")
        zs, ys = z[m], y[m]
        points = int(min(max(points, 10), 300))
        edges = np.linspace(0, zs.size, points + 1).astype(int)
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            rows.append({"km": round(float(zs[a:b].mean()), 3), "mean_db": round(float(ys[a:b].mean()), 2),
                         "peak_db": round(float(ys[a:b].max()), 2)})
        return {"wavelength_nm": wl, "from_site": site, "pulse_width_ns": tr["settings"]["pulse_ns"],
                "note": "Backscatter level (dB, one-way display). peak_db shows reflections inside each bin.",
                "samples": rows}

    def otdr_trace_image(self, wavelength_nm: int, from_site: str, start_km: float = None,
                         end_km: float = None) -> dict:
        wl, site = self._check_otdr(wavelength_nm, from_site)
        tr = otdr.synthesize(self.case, wl, site)
        tag = f"otdr_{wl}_{site}_{start_km}_{end_km}".replace(".", "p")
        path = images.render_trace(tr, f"OTDR {wl} nm from site {site}",
                                   os.path.join(self.workdir, tag + ".png"), start_km, end_km)
        return {"image_path": path, "wavelength_nm": wl, "from_site": site}

    def inspect_connector(self, element_id: str) -> dict:
        el = self._element(element_id)
        if el.kind != "connector":
            raise ToolError(f"{element_id} is a splice closure; only connectors can be inspected")
        path = images.render_endface(el.endface, self.case.seed * 7 + CONNECTOR_IDS.index(el.id),
                                     os.path.join(self.workdir, f"endface_{el.id}.png"))
        return {"image_path": path, "element_id": el.id,
                "note": "Inspection scope view of the bulkhead-side connector end face (core at centre, cladding disc, ferrule)."}

    def measure_power(self, point: str) -> dict:
        if point not in POWER_POINTS:
            raise ToolError(f"point must be one of {list(POWER_POINTS)}")
        c = self.case
        rng = np.random.default_rng([c.seed, 99, POWER_POINTS.index(point), len(self.calls)])
        tx_now = float(self._hist["tx_power_dbm"][-1] - 0.2)
        if point == "A-TX":
            v = tx_now
        else:
            if c.break_fibre_km is not None:
                return {"point": point, "wavelength_nm": 1550, "power_dbm": None, "reading": "no light (below -50 dBm)"}
            v = tx_now - link_loss(c, 1550, True)
            if point == "B-RX":
                v -= c.att_installed_db + 0.1
        return {"point": point, "wavelength_nm": 1550, "power_dbm": round(v + float(rng.normal(0, 0.05)), 2),
                "note": {"A-TX": "transmitter output at site A patch cord",
                         "B-ODF": "power arriving at site B ODF, before the attenuator",
                         "B-RX": "power at the receiver input, after the attenuator"}[point]}

    def get_site_notes(self) -> dict:
        return {"notes": self.case.notes}

    # ------------------------------------------------------------ helpers
    def _check_otdr(self, wavelength_nm, from_site):
        try:
            wl = int(wavelength_nm)
        except (TypeError, ValueError):
            raise ToolError("wavelength_nm must be 1310 or 1550")
        if wl not in (1310, 1550):
            raise ToolError("wavelength_nm must be 1310 or 1550")
        site = str(from_site).upper().replace("SITE ", "").strip()
        if site not in ("A", "B"):
            raise ToolError("from_site must be 'A' or 'B'")
        return wl, site

    def _element(self, element_id: str):
        for e in self.case.elements:
            if e.id == element_id:
                return e
        raise ToolError(f"unknown element_id {element_id}")

    def call(self, name: str, args: dict) -> dict:
        if name not in self.names:
            raise ToolError(f"unknown tool {name}")
        fn = getattr(self, name)
        try:
            out = fn(**(args or {}))
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}")
        self.calls.append({"tool": name, "args": args})
        return out


TOOL_SPECS = [
    {"name": "get_link_record", "description": "As-built record of the link: route map with element ids and route km, cable data, service and equipment specs, design attenuator, commissioning loss and alarm thresholds.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_alarms", "description": "Alarm log of the 1550 nm service receiver at site B over the last 14 days.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_pm", "description": "Performance monitoring history of the service. Daily min/avg/max for 14 days, or hourly values for the last N hours (max 72).",
     "parameters": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": list(pm.METRICS)},
         "resolution": {"type": "string", "enum": ["daily", "hourly"]},
         "last_hours": {"type": "integer", "description": "hourly only, 1 to 72"}}, "required": ["metric"]}},
    {"name": "otdr_event_table", "description": "Run an OTDR measurement (traffic is taken down for the test) and return the automatic event table.",
     "parameters": {"type": "object", "properties": {
         "wavelength_nm": {"type": "integer", "enum": [1310, 1550]},
         "from_site": {"type": "string", "enum": ["A", "B"]}}, "required": ["wavelength_nm", "from_site"]}},
    {"name": "otdr_trace", "description": "Downsampled OTDR backscatter trace (mean and peak level per bin) for a distance window.",
     "parameters": {"type": "object", "properties": {
         "wavelength_nm": {"type": "integer", "enum": [1310, 1550]},
         "from_site": {"type": "string", "enum": ["A", "B"]},
         "start_km": {"type": "number"}, "end_km": {"type": "number"},
         "points": {"type": "integer", "description": "number of bins, 10 to 300"}}, "required": ["wavelength_nm", "from_site"]}},
    {"name": "otdr_trace_image", "description": "Plot of the OTDR trace as an image, optionally zoomed to a distance window.",
     "parameters": {"type": "object", "properties": {
         "wavelength_nm": {"type": "integer", "enum": [1310, 1550]},
         "from_site": {"type": "string", "enum": ["A", "B"]},
         "start_km": {"type": "number"}, "end_km": {"type": "number"}}, "required": ["wavelength_nm", "from_site"]}},
    {"name": "inspect_connector", "description": "Inspection-scope image of a connector end face. Valid for connector elements only (A-ODF, MID-PP if present, B-ODF).",
     "parameters": {"type": "object", "properties": {"element_id": {"type": "string"}}, "required": ["element_id"]}},
    {"name": "measure_power", "description": "Optical power meter reading at 1550 nm at a test point: A-TX (transmitter output), B-ODF (arriving at site B, before the attenuator) or B-RX (receiver input).",
     "parameters": {"type": "object", "properties": {"point": {"type": "string", "enum": list(POWER_POINTS)}}, "required": ["point"]}},
    {"name": "get_site_notes", "description": "Field engineering log entries for the link over the last two weeks.",
     "parameters": {"type": "object", "properties": {}}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SPECS}

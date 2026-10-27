"""Seeded simulator for point-to-point fibre links with one injected fault.

Each case is a single-mode fibre link from site A to site B carrying one
1550 nm service (transmitter at A, fixed attenuator and receiver at B).
The physical model covers:

* fibre attenuation at 1310 and 1550 nm, cable excess length (helix) and
  slack loops stored at splice closures, so OTDR (fibre) distance differs
  from route distance on the as-built map;
* connectors, splices (with mode-field-diameter mismatch, which makes the
  same splice look like a loss from one end and a gain from the other) and
  an optional street-cabinet patch panel;
* seven fault types with their physical signatures (see FAULT_TYPES).

The truth lives in ``Case.truth``; everything an agent may see is produced
by ``fibresight.tools.ToolBox`` from the hidden ``Case`` object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

FAULT_TYPES = [
    "dirty_connector",
    "damaged_connector",
    "macrobend",
    "bad_splice",
    "fibre_break",
    "wrong_attenuator",
    "tx_degradation",
]

ACTION_FOR_FAULT = {
    "dirty_connector": "clean_connector",
    "damaged_connector": "replace_connector",
    "macrobend": "relieve_bend",
    "bad_splice": "resplice",
    "fibre_break": "repair_break",
    "wrong_attenuator": "replace_attenuator",
    "tx_degradation": "replace_transmitter",
}

# Faults whose location is graded by element id; the others by route km.
ELEMENT_FAULTS = {"dirty_connector", "damaged_connector", "bad_splice",
                  "wrong_attenuator", "tx_degradation"}
DISTRIBUTED_FAULTS = {"macrobend", "fibre_break"}

WAVELENGTHS = (1310, 1550)
ATTENUATOR_VALUES = (0, 3, 5, 7, 10, 15)

# Receiver and alarm thresholds (dBm) for the 1550 nm service.
RX_SENSITIVITY = -15.0
RX_OVERLOAD = -1.0
RX_LOW_WARN = -13.0
RX_LOW_ALARM = -16.0
RX_LOS = -25.0
RX_HIGH_WARN = -3.0
RX_HIGH_ALARM = -1.0
BER_ALARM = 1e-6
TX_SPEC = (-1.0, 2.0)
NOW = datetime(2026, 10, 27, 8, 0, tzinfo=timezone.utc)
PM_DAYS = 14


@dataclass
class Element:
    id: str
    kind: str                 # "connector" | "splice"
    route_km: float
    fibre_km: float = 0.0     # distance along the fibre from site A
    slack_m: float = 0.0      # slack stored at a closure (splices only)
    loss: dict = field(default_factory=dict)        # true loss per wavelength
    mfd_offset: dict = field(default_factory=lambda: {1310: 0.0, 1550: 0.0})  # apparent extra loss from A (gain from B)
    note: str = ""
    reflectance: Optional[float] = None             # dB, connectors only
    endface: str = "clean"    # "clean" | "dirty" | "damaged" (connectors)


@dataclass
class Case:
    seed: int
    route_km: float
    excess_length: float      # cable helix factor, e.g. 0.012
    alpha: dict               # dB/km per wavelength
    elements: list
    tx_dbm: float
    att_design_db: float
    att_installed_db: float
    fault: dict               # truth and fault parameters
    events: list = field(default_factory=list)      # extra point events
    break_fibre_km: Optional[float] = None
    break_reflectance: Optional[float] = None
    onset: Optional[datetime] = None
    notes: list = field(default_factory=list)
    commissioning_loss: dict = field(default_factory=dict)
    tx_bias0: float = 40.0
    tx_eol: dict = field(default_factory=dict)

    @property
    def fibre_length_km(self) -> float:
        return self.elements[-1].fibre_km

    @property
    def truth(self) -> dict:
        return {k: self.fault[k] for k in ("fault_type", "element_id", "route_km", "action")}

    def route_to_fibre(self, route_km: float) -> float:
        """Fibre distance from A of a point at a given route distance (mid-span)."""
        slack = sum(e.slack_m for e in self.elements
                    if e.kind == "splice" and e.route_km < route_km) / 1000.0
        return route_km * (1.0 + self.excess_length) + slack


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([20261027, seed]))


def _build_route(rng: np.random.Generator) -> tuple:
    route = float(np.round(rng.uniform(14.0, 52.0), 2))
    elements = [Element("A-ODF", "connector", 0.0)]
    pos, n = 0.0, 0
    while True:
        pos += rng.uniform(2.0, 6.0)
        if pos > route - 1.5:
            break
        n += 1
        elements.append(Element(f"SC-{n:02d}", "splice", float(np.round(pos, 3))))
    if rng.random() < 0.6:
        # Street cabinet patch panel away from closures.
        for _ in range(50):
            r = rng.uniform(0.25 * route, 0.75 * route)
            if all(abs(r - e.route_km) > 0.8 for e in elements):
                elements.append(Element("MID-PP", "connector", float(np.round(r, 3))))
                break
    elements.append(Element("B-ODF", "connector", route))
    elements.sort(key=lambda e: e.route_km)
    return route, elements


def _place_on_fibre(case: Case) -> None:
    slack_before = 0.0
    for e in case.elements:
        base = e.route_km * (1.0 + case.excess_length) + slack_before / 1000.0
        if e.kind == "splice":
            e.fibre_km = base + e.slack_m / 2000.0
            slack_before += e.slack_m
        else:
            e.fibre_km = base


def _healthy_parameters(case: Case, rng: np.random.Generator) -> None:
    for e in case.elements:
        if e.kind == "connector":
            l = rng.uniform(0.15, 0.35)
            e.loss = {1310: l + rng.uniform(-0.02, 0.02), 1550: l}
            e.reflectance = rng.uniform(-56.0, -48.0)
        else:
            e.slack_m = float(np.round(rng.uniform(30.0, 80.0), 1))
            l = rng.uniform(0.01, 0.08)
            e.loss = {1310: l, 1550: l}
            big = rng.random() < 0.3
            m = rng.uniform(-0.28, 0.28) if big else rng.uniform(-0.05, 0.05)
            e.mfd_offset = {1310: m * rng.uniform(0.7, 0.9), 1550: m}
    splices = [e for e in case.elements if e.kind == "splice"]
    if splices and rng.random() < 0.55:
        # One closure joins a G.655 (non-zero dispersion shifted) section; the
        # mode-field mismatch makes it look like a big, wavelength-dependent
        # loss from one end and a gainer from the other.
        sc = splices[int(rng.integers(len(splices)))]
        m = rng.choice([-1.0, 1.0]) * rng.uniform(0.55, 0.9)
        sc.mfd_offset = {1550: m, 1310: m * rng.uniform(0.4, 0.6)}
        sc.note = "joins the original G.652.D cable to a G.655 replacement section"


def link_loss(case: Case, wl: int, include_fault: bool = True) -> float:
    """End-to-end optical loss A-ODF to B-ODF (both connectors included)."""
    total = case.alpha[wl] * case.fibre_length_km
    total += sum(e.loss[wl] for e in case.elements)
    if include_fault:
        total += sum(ev["loss"][wl] for ev in case.events)
    return total


def _choose_attenuator(rx_no_att: float) -> float:
    target = -11.9
    best = min(ATTENUATOR_VALUES, key=lambda a: abs(rx_no_att - a - target))
    return float(best)


def _inject_fault(case: Case, fault_type: str, rng: np.random.Generator) -> None:
    connectors = [e for e in case.elements if e.kind == "connector"]
    splices = [e for e in case.elements if e.kind == "splice"]
    healthy_rx = case.tx_dbm - link_loss(case, 1550, False) - case.att_design_db
    # Extra loss large enough to cross the Rx low warning.
    need = healthy_rx - (RX_LOW_WARN - 0.3)
    extra = max(rng.uniform(0.9, 3.5), need)
    fault = {"fault_type": fault_type, "action": ACTION_FOR_FAULT[fault_type],
             "element_id": None, "route_km": None}

    if fault_type in ("dirty_connector", "damaged_connector"):
        # The far-end B-ODF is the OTDR end point from A; keep faults on
        # connectors that both OTDR directions can characterise.
        el = connectors[int(rng.integers(len(connectors)))]
        el.loss = {1550: el.loss[1550] + extra,
                   1310: el.loss[1310] + extra * rng.uniform(0.9, 1.1)}
        if fault_type == "dirty_connector":
            el.reflectance = rng.uniform(-36.0, -28.0)
            el.endface = "dirty"
        else:
            el.reflectance = rng.uniform(-30.0, -20.0)
            el.endface = "damaged"
        fault.update(element_id=el.id, route_km=el.route_km, extra_1550=extra)
    elif fault_type == "bad_splice":
        el = splices[int(rng.integers(len(splices)))]
        l1310 = max(rng.uniform(0.9, 3.0), need)
        el.loss = {1310: el.loss[1310] + l1310,
                   1550: el.loss[1550] + l1310 * rng.uniform(0.85, 1.15)}
        before = el.loss[1310] - l1310  # healthy splice loss (same at both wavelengths)
        if el.loss[1550] - before < need:
            el.loss[1550] = before + need
        el.mfd_offset = {1310: 0.0, 1550: 0.0}
        fault.update(element_id=el.id, route_km=el.route_km, extra_1550=el.loss[1550] - before)
    elif fault_type == "macrobend":
        l1550 = extra
        ratio = rng.uniform(0.08, 0.25)
        candidates = [s for s in splices if s.route_km >= 8.0]
        if candidates and rng.random() < 0.5:
            sc = candidates[int(rng.integers(len(candidates)))]
            fibre = sc.fibre_km + rng.uniform(-0.008, 0.008)
            route = sc.route_km
            fault["element_id"] = sc.id
        else:
            route = _midspan_route(case, rng, min_km=8.0)
            fibre = case.route_to_fibre(route)
        case.events.append({"kind": "bend", "fibre_km": fibre,
                            "loss": {1550: l1550, 1310: l1550 * ratio}})
        fault["route_km"] = float(np.round(route, 3))
        fault["extra_1550"] = l1550
    elif fault_type == "fibre_break":
        route = _midspan_route(case, rng, min_km=8.0)
        case.break_fibre_km = case.route_to_fibre(route)
        case.break_reflectance = (rng.uniform(-30.0, -14.0) if rng.random() < 0.5 else None)
        fault["route_km"] = float(np.round(route, 3))
    elif fault_type == "wrong_attenuator":
        rx_no_att = case.tx_dbm - link_loss(case, 1550, False)
        options = []
        for a in ATTENUATOR_VALUES:
            rx = rx_no_att - a
            if abs(a - case.att_design_db) >= 5 and (rx > RX_HIGH_WARN + 0.5 or rx < RX_LOW_ALARM - 0.5):
                options.append(a)
        if not options:
            options = [case.att_design_db + 10.0]
        case.att_installed_db = float(options[int(rng.integers(len(options)))])
        fault.update(element_id="B-ATT", route_km=case.route_km)
    elif fault_type == "tx_degradation":
        # Laser end of life: APC raises bias until it saturates, then power falls.
        drop = max(rng.uniform(2.5, 4.5), need)
        days_rise = rng.uniform(9.0, 13.0)
        days_fall = rng.uniform(2.0, 4.5)
        case.tx_eol = {"bias_rise": rng.uniform(25.0, 35.0), "days_rise": days_rise,
                       "days_fall": days_fall, "drop": drop}
        fault.update(element_id="A-TX", route_km=0.0)
    case.fault = fault


def _midspan_route(case: Case, rng: np.random.Generator, min_km: float) -> float:
    for _ in range(500):
        r = rng.uniform(min_km, case.route_km - 1.0)
        if all(abs(r - e.route_km) > 0.4 for e in case.elements):
            return float(np.round(r, 3))
    return float(np.round(0.5 * (case.route_km + min_km), 3))


def _onset(case: Case, rng: np.random.Generator) -> None:
    ft = case.fault["fault_type"]
    if ft == "tx_degradation":
        case.onset = NOW - timedelta(days=case.tx_eol["days_rise"] + case.tx_eol["days_fall"])
    else:
        days_ago = rng.uniform(1.5, 9.5)
        case.onset = (NOW - timedelta(days=days_ago)).replace(minute=0, second=0, microsecond=0)


def _notes(case: Case, rng: np.random.Generator) -> None:
    """Field log: one entry tied to the fault onset (when there is a work
    event behind it) plus routine and misleading entries."""
    ft, el = case.fault["fault_type"], case.fault["element_id"]
    notes = []
    splices = [e for e in case.elements if e.kind == "splice"]
    connectors = [e for e in case.elements if e.kind == "connector"]

    def at(days_ago: float) -> datetime:
        t = NOW - timedelta(days=days_ago)
        return t.replace(minute=int(rng.integers(0, 4)) * 15, second=0, microsecond=0)

    onset_days = (NOW - case.onset).total_seconds() / 86400.0
    t_on = case.onset - timedelta(minutes=int(rng.integers(20, 80)))
    if ft in ("bad_splice",) or (ft == "macrobend" and el):
        notes.append((t_on, f"Splicing crew re-entered closure {el} to add a branch cable for a new customer. Closure resealed."))
    elif ft in ("dirty_connector", "damaged_connector"):
        where = {"A-ODF": "site A ODF", "B-ODF": "site B ODF", "MID-PP": "street cabinet patch panel"}[el]
        notes.append((t_on, f"Patching work at {where}: jumpers moved to free a port for a new service."))
        if ft == "damaged_connector":
            notes.append((at(max(onset_days - 1.0, 0.3)),
                          f"Connector at {where} cleaned with click cleaner after low Rx reported. No change in Rx power."))
    elif ft == "wrong_attenuator":
        notes.append((t_on, "Receiver module at site B replaced under RMA. Attenuator refitted from spares box."))
    elif ft == "macrobend":
        notes.append((at(onset_days + rng.uniform(0.2, 1.0)),
                      "Utility contractor reported working in ducts near the route. No damage reported."))
    elif ft == "fibre_break":
        r = case.fault["route_km"]
        approx = r + rng.uniform(-2.5, 2.5)
        notes.append((t_on + timedelta(minutes=50),
                      f"Customer reports total outage. Road works reported somewhere around route km {approx:.0f}."))

    # Misleading or routine entries.
    decoys = []
    other_sc = [s for s in splices if s.id != el]
    if other_sc:
        sc = other_sc[int(rng.integers(len(other_sc)))]
        decoys.append(f"Splicing crew re-entered closure {sc.id} to add a branch cable for a new customer. Closure resealed.")
    other_conn = [c for c in connectors if c.id != el]
    c = other_conn[int(rng.integers(len(other_conn)))]
    where = {"A-ODF": "site A ODF", "B-ODF": "site B ODF", "MID-PP": "street cabinet patch panel"}[c.id]
    decoys.append(f"Connector at {where} inspected and cleaned during routine maintenance. Passed inspection.")
    decoys.append(f"Heavy rain and flooding reported along the route near km {rng.uniform(3, case.route_km - 2):.0f}.")
    decoys.append("Power works at site B; equipment ran on battery for 40 minutes, no traffic impact.")
    rng.shuffle(decoys)
    for text in decoys[: int(rng.integers(2, 4))]:
        days = rng.uniform(0.5, PM_DAYS - 0.5)
        if abs(days - onset_days) < 1.0:
            days = min(PM_DAYS - 0.5, days + 2.0)
        notes.append((at(days), text))
    notes.sort(key=lambda x: x[0])
    case.notes = [{"time": t.strftime("%Y-%m-%d %H:%M UTC"), "entry": s} for t, s in notes]


def generate_case(seed: int, fault_type: Optional[str] = None) -> Case:
    rng = _rng(seed)
    if fault_type is None:
        fault_type = FAULT_TYPES[seed % len(FAULT_TYPES)]
    for _ in range(50):
        route, elements = _build_route(rng)
        alpha1550 = rng.uniform(0.185, 0.21)
        case = Case(
            seed=seed, route_km=route, excess_length=float(rng.uniform(0.010, 0.018)),
            alpha={1550: alpha1550, 1310: alpha1550 + rng.uniform(0.13, 0.15)},
            elements=elements, tx_dbm=float(rng.uniform(0.0, 1.8)),
            att_design_db=0.0, att_installed_db=0.0, fault={},
            tx_bias0=float(rng.uniform(34.0, 46.0)),
        )
        _healthy_parameters(case, rng)
        _place_on_fibre(case)
        case.att_design_db = _choose_attenuator(case.tx_dbm - link_loss(case, 1550, False))
        healthy = case.tx_dbm - link_loss(case, 1550, False) - case.att_design_db
        if healthy < -12.5:  # long link: a stronger transmitter was fitted at build
            case.tx_dbm = float(min(2.0, case.tx_dbm + (-12.0 - healthy)))
            healthy = case.tx_dbm - link_loss(case, 1550, False) - case.att_design_db
        if -12.5 <= healthy <= -10.0:
            break  # otherwise the planner would have redesigned the link; draw again
    case.commissioning_loss = {wl: round(link_loss(case, wl, False) + rng.normal(0, 0.05), 2)
                               for wl in WAVELENGTHS}
    case.att_installed_db = case.att_design_db
    _inject_fault(case, fault_type, rng)
    _onset(case, rng)
    _notes(case, rng)
    return case

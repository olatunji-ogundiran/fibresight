"""Simulate the technician's repair, so a fix can be checked by measurement.

Only the right action at the right place removes the fault. Cleaning a
damaged connector, re-splicing the wrong closure or digging 2 km from the
break leaves the link as it was, and the next power reading shows it.
"""
from __future__ import annotations

import copy

from .sim import Case

ROUTE_MATCH_KM = 0.10  # a crew working at the reported spot finds a fault this close


def fix_works(case: Case, action: str, element_id: str = None, route_km: float = None) -> bool:
    t = case.truth
    ft, eid = t["fault_type"], t["element_id"]
    if action == "clean_connector":
        return ft == "dirty_connector" and element_id == eid
    if action == "replace_connector":
        return ft in ("dirty_connector", "damaged_connector") and element_id == eid
    if action == "resplice":
        return ft == "bad_splice" and element_id == eid
    if action in ("relieve_bend", "repair_break"):
        if ft != ("macrobend" if action == "relieve_bend" else "fibre_break"):
            return False
        if eid and element_id == eid:  # bend inside a closure
            return True
        return route_km is not None and abs(float(route_km) - t["route_km"]) <= ROUTE_MATCH_KM
    if action == "replace_attenuator":
        return ft == "wrong_attenuator"
    if action == "replace_transmitter":
        return ft == "tx_degradation"
    return False


def apply_fix(case: Case, action: str, element_id: str = None, route_km: float = None) -> tuple:
    """Return (case after the work, fixed?). The truth record is unchanged."""
    if getattr(case, "repaired", False) or not fix_works(case, action, element_id, route_km):
        return case, bool(getattr(case, "repaired", False))
    new = copy.deepcopy(case)
    h = case._healthy
    ft = case.truth["fault_type"]
    if ft in ("dirty_connector", "damaged_connector", "bad_splice"):
        el = next(e for e in new.elements if e.id == case.truth["element_id"])
        old = next(e for e in h.elements if e.id == el.id)
        el.loss = {**old.loss, 1625: old.loss[1550] + 0.01}
        el.reflectance, el.endface = old.reflectance, old.endface
        el.mfd_offset = {**old.mfd_offset, 1625: old.mfd_offset.get(1550, 0.0) * 1.1}
    elif ft == "macrobend":
        new.events = [ev for ev in new.events if ev["kind"] != "bend"]
    elif ft == "fibre_break":
        new.break_fibre_km = None
    elif ft == "wrong_attenuator":
        new.att_installed_db = new.att_design_db
    elif ft == "tx_degradation":
        new.tx_eol = {}
    new.repaired = True
    return new, True

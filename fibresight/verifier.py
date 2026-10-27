"""Strict, deterministic grading of a diagnosis against the hidden truth.

A diagnosis passes only if the fault type, the corrective action and the
location are all right. Location is graded by element id for faults that
sit at an element, and by route distance (km on the as-built route map,
within ROUTE_TOLERANCE_KM) for faults that can sit anywhere along the
cable. Malformed answers, lists of guesses and non-finite numbers score 0.
"""
from __future__ import annotations

import math

from .sim import ACTION_FOR_FAULT, DISTRIBUTED_FAULTS, FAULT_TYPES, Case

ROUTE_TOLERANCE_KM = 0.10
ANSWER_KEYS = {"fault_type", "element_id", "route_km", "action", "evidence"}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "fault_type": {"type": "string", "enum": FAULT_TYPES},
        "element_id": {"type": ["string", "null"],
                       "description": "Element or equipment id from the link record (e.g. SC-03, MID-PP, B-ATT, A-TX). Required for faults at an element."},
        "route_km": {"type": ["number", "null"],
                     "description": "Distance from site A along the as-built route map in km. Required for macrobend and fibre_break."},
        "action": {"type": "string", "enum": sorted(set(ACTION_FOR_FAULT.values()))},
        "evidence": {"type": "string", "description": "Short justification citing the measurements used."},
    },
    "required": ["fault_type", "action", "evidence"],
}


def validate(answer) -> list:
    errs = []
    if not isinstance(answer, dict):
        return ["answer must be a single JSON object"]
    extra = set(answer) - ANSWER_KEYS
    if extra:
        errs.append(f"unexpected keys {sorted(extra)}")
    ft = answer.get("fault_type")
    if not isinstance(ft, str) or ft not in FAULT_TYPES:
        errs.append("fault_type must be one of " + ", ".join(FAULT_TYPES))
    act = answer.get("action")
    if not isinstance(act, str) or act not in ACTION_FOR_FAULT.values():
        errs.append("action must be one of " + ", ".join(sorted(set(ACTION_FOR_FAULT.values()))))
    eid = answer.get("element_id")
    if eid is not None and not isinstance(eid, str):
        errs.append("element_id must be a string or null")
    rk = answer.get("route_km")
    if rk is not None:
        if isinstance(rk, bool) or not isinstance(rk, (int, float)) or not math.isfinite(rk):
            errs.append("route_km must be a finite number or null")
    if isinstance(ft, str) and ft in DISTRIBUTED_FAULTS and rk is None:
        errs.append(f"route_km is required for {ft}")
    if isinstance(ft, str) and ft in FAULT_TYPES and ft not in DISTRIBUTED_FAULTS and not eid:
        errs.append(f"element_id is required for {ft}")
    return errs


def grade(answer, case: Case) -> dict:
    truth = case.truth
    errs = validate(answer)
    if errs:
        return {"score": 0.0, "valid": False, "errors": errs, "fault_type_ok": False,
                "action_ok": False, "location_ok": False, "truth": truth}
    ft_ok = answer["fault_type"] == truth["fault_type"]
    act_ok = answer["action"] == truth["action"]
    if truth["fault_type"] in DISTRIBUTED_FAULTS:
        loc_err = abs(float(answer["route_km"]) - truth["route_km"]) if answer.get("route_km") is not None else None
        loc_ok = loc_err is not None and loc_err <= ROUTE_TOLERANCE_KM
    else:
        loc_err = None
        loc_ok = answer.get("element_id") == truth["element_id"]
    passed = ft_ok and act_ok and loc_ok
    return {"score": 1.0 if passed else 0.0, "valid": True, "errors": [], "fault_type_ok": ft_ok,
            "action_ok": act_ok, "location_ok": loc_ok,
            "route_error_km": None if loc_err is None else round(loc_err, 3), "truth": truth}

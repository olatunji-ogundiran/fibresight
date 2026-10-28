"""Turn a diagnosis into a dispatch work order for the NOC.

Deterministic on purpose: the agent decides what is wrong and where; this
module states who to send, what they carry and where exactly to go, using
only the link record, so the order is consistent and checkable.
"""
from __future__ import annotations

CREW = {
    "dirty_connector": ("Patching technician", ["Fibre inspection scope", "Click cleaner and lint-free wipes", "Optical power meter"]),
    "damaged_connector": ("Patching technician", ["Fibre inspection scope", "Replacement patch cord / pigtail (SC/UPC)", "Optical power meter", "Click cleaner"]),
    "macrobend": ("Fibre maintenance crew", ["OTDR", "Visual fault locator", "Closure re-entry kit", "Cable ties and fibre management tray"]),
    "bad_splice": ("Splicing crew", ["Fusion splicer and cleaver", "Splice protection sleeves", "OTDR", "Closure re-entry kit"]),
    "fibre_break": ("Fibre restoration crew", ["OTDR", "Fusion splicer and cleaver", "Repair closure and spare cable", "Excavation tools", "Visual fault locator"]),
    "wrong_attenuator": ("Equipment technician", ["Fixed attenuator set (3, 5, 7, 10, 15 dB)", "Optical power meter", "Inspection scope and cleaner"]),
    "tx_degradation": ("Equipment technician", ["Spare 10G 1550 nm transceiver", "Optical power meter", "Inspection scope and cleaner", "ESD strap"]),
}

PRIORITY = {"fibre_break": "P1, service down", "tx_degradation": "P2, degrading", "wrong_attenuator": "P2, errors on service"}


def _where(answer: dict, record: dict) -> str:
    elems = record["elements_in_route_order"]
    eid, km = answer.get("element_id"), answer.get("route_km")
    names = {"A-ODF": "ODF at site A", "B-ODF": "ODF at site B", "MID-PP": "street cabinet patch panel (MID-PP)",
             "A-TX": "transmitter at site A", "B-ATT": "in-line attenuator at site B, between ODF and receiver"}
    if eid and answer.get("fault_type") not in ("macrobend", "fibre_break"):
        if eid in names:
            return names[eid]
        el = next((e for e in elems if e["id"] == eid), None)
        return f"splice closure {eid} at route km {el['route_km']:.3f}" if el else eid
    if km is None:
        return "location not given"
    km = float(km)
    before = max((e for e in elems if e["route_km"] <= km), key=lambda e: e["route_km"], default=None)
    after = min((e for e in elems if e["route_km"] > km), key=lambda e: e["route_km"], default=None)
    at = next((e for e in elems if abs(e["route_km"] - km) < 0.02), None)
    if at:
        return f"at {at['id']} (route km {km:.3f}), inside the closure / cabinet"
    parts = [f"route km {km:.3f} from site A"]
    if before:
        parts.append(f"{(km - before['route_km']) * 1000:.0f} m past {before['id']}")
    if after:
        parts.append(f"{(after['route_km'] - km) * 1000:.0f} m before {after['id']}")
    return ", ".join(parts)


def work_order(answer: dict, record: dict, ticket_ref: str = "FS-0001") -> dict:
    if not answer:
        return {}
    ft = answer.get("fault_type")
    crew, tools = CREW.get(ft, ("Field technician", ["OTDR", "Power meter"]))
    return {
        "ticket": ticket_ref,
        "priority": PRIORITY.get(ft, "P3, degraded service"),
        "fault": ft,
        "location": _where(answer, record),
        "send": crew,
        "carry": tools,
        "action": answer.get("action"),
        "verify_after_fix": "Rx power back within design range (about -10 to -12.5 dBm) and pre-FEC BER below 1e-9; clear alarms; re-run OTDR from both ends if fibre was touched.",
        "evidence": answer.get("evidence", ""),
    }

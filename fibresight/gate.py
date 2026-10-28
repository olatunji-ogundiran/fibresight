"""Runtime verification: checks that run inside submit_diagnosis, before a
diagnosis is accepted. Unlike the verifier (which grades against the
simulator's hidden truth for evaluation), everything here uses only what is
available in a real deployment: the answer, the tool outputs and the link
record. So the same gate works on a live network.

Layer 1, deterministic (always on): the answer is well formed, names a real
element or a route distance on the map, uses an action that fixes that fault,
cites at least two measurements that match the tool outputs they cite, and
leaves no checklist item open.

Layer 2, reviewer (optional): a second Meta model reads the diagnosis, the
cited tool outputs and every other measurement taken, and approves or rejects
it with reasons.

A rejected submission goes back to the agent with the problems listed. After
MAX_ATTEMPTS rejections the ticket is escalated to a human engineer.
"""
from __future__ import annotations

import json
import re

from .sim import ACTION_FOR_FAULT
from .verifier import grounding, validate

MAX_ATTEMPTS = 3
MIN_CITATIONS = 2
CONNECTOR_FAULTS = ("dirty_connector", "damaged_connector")


def deterministic_checks(answer, tb) -> list:
    if not isinstance(answer, dict):
        return ["the diagnosis must be a single JSON object"]
    problems = list(validate(answer))
    record = tb.get_link_record()
    valid_ids = set(record["answer_ids"]["elements"]) | set(record["answer_ids"]["equipment"])
    route_len = record["elements_in_route_order"][-1]["route_km"]
    eid, rk, ft, act = answer.get("element_id"), answer.get("route_km"), answer.get("fault_type"), answer.get("action")
    if isinstance(eid, str) and eid and eid not in valid_ids:
        problems.append(f"element_id '{eid}' is not on the link record (valid: {', '.join(sorted(valid_ids))})")
    if isinstance(rk, (int, float)) and not isinstance(rk, bool) and not 0.0 <= rk <= route_len:
        problems.append(f"route_km {rk} is outside the route (0 to {route_len:.3f} km)")
    if ft in ACTION_FOR_FAULT and isinstance(act, str):
        ok = {"clean_connector", "replace_connector"} if ft in CONNECTOR_FAULTS else {ACTION_FOR_FAULT[ft]}
        if act not in ok:
            problems.append(f"action '{act}' does not fix a {ft}; use {' or '.join(sorted(ok))}")
    refs = answer.get("evidence_refs") or []
    if not isinstance(refs, list) or len(refs) < MIN_CITATIONS:
        problems.append(f"cite at least {MIN_CITATIONS} key measurements in evidence_refs (call_id and the exact value quoted)")
    else:
        g = grounding(answer, tb.calls)
        for r in g["ungrounded"]:
            problems.append(f"citation does not match the output of the call it cites: {json.dumps(r)[:200]}")
    if not tb.checklist:
        problems.append("no plan: write the checklist with set_checklist and close each item")
    else:
        still_open = [i["item"] for i in tb.checklist if i["status"] == "open"]
        if still_open:
            problems.append(f"{len(still_open)} checklist item(s) still open: " + "; ".join(still_open[:4])
                            + ". Mark each done, skipped or failed with a note.")
    return problems


MEASUREMENTS = ("get_alarms", "get_pm", "embedded_otdr", "otdr_event_table", "otdr_trace", "measure_power",
                "segment_endface", "inspect_connector", "get_site_notes")
REVIEW_CHARS = 30000


REVIEW_PROMPT = """You are a senior transmission engineer reviewing a colleague's diagnosis of a fibre link fault
before a crew is dispatched. You see the diagnosis, the measurements it cites, every other measurement taken
during the investigation (look at all of it, not only what was cited) and the route map. Decide whether the cited evidence supports the stated fault type, the
location and the action. Approve when it does. Reject only for a concrete problem: a contradiction with the data,
a location that does not follow from the measurements, or a key alternative cause the evidence does not rule out.
Do not reject for style. Reply with JSON only: {"verdict": "approve" or "reject", "reasons": ["short reason", ...]}"""

REMOTE_NOTE = """
This is a remote NOC diagnosis: nobody is on site yet and the crew confirms on arrival. Some things cannot be measured
remotely (a connector's end face, an attenuator's value, a patch cord at the receiver). Approve when the remote evidence
points to one cause and location and reasonably rules out the others; do not demand proof only a site visit can give."""


def review(client, model: str, answer: dict, tb) -> dict:
    """Second-model check of the claims. Errors never block a submission:
    they are reported as 'not reviewed'."""
    record = tb.get_link_record()
    cited_ids = set()
    for r in answer.get("evidence_refs") or []:
        try:
            cited_ids.add(int(r["call_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    # Cited outputs first, then every other measurement, within a size budget.
    order = sorted(range(1, len(tb.calls) + 1), key=lambda i: (i not in cited_ids, i))
    outputs, used = {}, 0
    for cid in order:
        c = tb.calls[cid - 1]
        if c["tool"] not in MEASUREMENTS and cid not in cited_ids:
            continue
        res = {k: v for k, v in c["result"].items() if k != "image_path"} if isinstance(c["result"], dict) else c["result"]
        out = json.dumps(res, default=str)[:4000]
        if used + len(out) > REVIEW_CHARS:
            continue
        used += len(out)
        outputs[cid] = {"tool": c["tool"], "args": c["args"], "cited": cid in cited_ids, "output": out}
    route = [(e["id"], e["route_km"]) for e in record["elements_in_route_order"]]
    content = json.dumps({"diagnosis": {k: answer.get(k) for k in ("fault_type", "element_id", "route_km", "action", "evidence")},
                          "citations": answer.get("evidence_refs"), "measurements": outputs,
                          "route_map": route, "service": record["service"],
                          "commissioning_link_loss_db": record["commissioning_link_loss_db"]}, default=str)
    try:
        resp = client.chat.completions.create(model=model, messages=[
            {"role": "system", "content": REVIEW_PROMPT + (REMOTE_NOTE if getattr(tb, "remote_only", False) else "")},
            {"role": "user", "content": content}])
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {}
        verdict = str(out.get("verdict", "")).lower()
        if verdict not in ("approve", "reject"):
            return {"verdict": "not reviewed", "reasons": ["reviewer reply was not valid JSON"], "model": model}
        return {"verdict": verdict, "reasons": [str(x) for x in (out.get("reasons") or [])][:5], "model": model}
    except Exception as e:  # provider or parsing error
        return {"verdict": "not reviewed", "reasons": [f"{type(e).__name__}: {str(e)[:120]}"], "model": model}

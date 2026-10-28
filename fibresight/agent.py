"""Tool-calling agent loop for any OpenAI-compatible endpoint (Baseten,
Modal, Meta Model API). The agent investigates one case with the ToolBox
and must finish by calling submit_diagnosis exactly once.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Optional

from .sim import Case
from .tools import TOOL_SPECS, ToolBox, ToolError
from .gate import MAX_ATTEMPTS, deterministic_checks, review
from .verifier import ANSWER_SCHEMA, grade, grade_dispatch, grounding

SUBMIT_SPEC = {"name": "submit_diagnosis",
               "description": "Submit the final diagnosis when you are confident. It is checked before it is accepted: it must be complete, cite at least two measurements that match the tool outputs, and every checklist item must be closed. A rejected submission lists the problems; fix them and submit again (3 attempts, then the ticket goes to a human engineer).",
               "parameters": ANSWER_SCHEMA}

BASE_PROMPT = """You are a transmission engineer on call for a fibre network operator.
A 1550 nm service from site A to site B is degraded or down. Investigate with the tools and find the single root cause.
Finish by calling submit_diagnosis with the fault type, its location and the corrective action.
Location: give element_id for faults at a connector, splice closure, attenuator (B-ATT) or transmitter (A-TX);
give route_km (distance from site A on the as-built route map) for a macrobend or a fibre break.
Start by writing your plan with set_checklist and keep it updated with update_checklist.
Every tool result carries a call_id. In submit_diagnosis, list the key measurements behind your diagnosis in
evidence_refs, each with the call_id it came from and the exact value quoted from that result."""

PLAYBOOK_PROMPT = BASE_PROMPT + """

Field procedure (follow it):
1. Read the link record, alarms, PM and site notes. Notes can be wrong or irrelevant; trust measurements.
2. Rule out equipment before chasing the fibre. Measure A-TX: compare it with the transmitter's own history,
   not just the spec; a laser at end of life shows bias current rising over days and output then falling.
   Measure B-ODF and B-RX: their difference is the installed attenuator; compare with the design value.
   Loss of signal at the receiver does not prove a break if light still arrives at B-ODF.
3. For fibre faults, shoot the OTDR from BOTH ends at BOTH 1310 and 1550 nm.
   - Average each event's loss over the two directions. Splices between different fibre types look like a big
     loss from one end and a gain from the other; the average is the true loss.
   - A reflective event with ~0 dB loss seen from one end only, at twice the distance of a strong reflection, is a ghost.
   - Reflective lossy event = connector: inspect its end face. Particles/film = dirty (clean it);
     scratches or chips through the core zone = damaged (replace it).
   - Non-reflective loss much larger at 1550 than at 1310 (ratio about 3 or more) = macrobend; similar at both = bad splice.
4. OTDR distance is FIBRE distance. Fibre is longer than the route because of cable excess length and slack loops
   stored at every splice closure. To get route km, find the splice-closure events in the trace, match them to the
   closures in the link record, and interpolate between the two closures either side of the fault.
5. Submit only one diagnosis."""


NOC_PROMPT = """You are a NOC transmission engineer for a fibre network operator, working remotely. Nobody is on site yet.
A 1550 nm service from site A to site B is degraded or down. Using only the remote tools (alarms, performance
monitoring, the embedded in-service OTDR at each site, the link record and the field log), find the single root cause
and exactly where it is, so the right crew can be sent to the right place with the right kit.
Finish by calling submit_diagnosis with the fault type, its location and the corrective action.
Location: give element_id for faults at a connector, splice closure, attenuator (B-ATT) or transmitter (A-TX);
give route_km (distance from site A on the as-built route map) for a macrobend or a fibre break.
Connector end faces cannot be inspected remotely. For a connector fault submit dirty_connector with clean_connector
unless the evidence shows it is damaged: the crew inspects, cleans, and replaces it if it is damaged.
Start by writing your plan with set_checklist and keep it updated with update_checklist.
Every tool result carries a call_id. In submit_diagnosis, list the key measurements behind your diagnosis in
evidence_refs, each with the call_id it came from and the exact value quoted from that result."""

NOC_TICKET = "NOC ticket: alarm raised on the A to B 1550 nm service. Diagnose it remotely so we can dispatch."


def _tool_list(tb):
    return [{"type": "function", "function": t} for t in tb.specs + [SUBMIT_SPEC]]


def _image_message(path: str, tool: str) -> dict:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"role": "user", "content": [
        {"type": "text", "text": f"Image returned by {tool}:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}


TICKET = "Ticket: customer reports problems on the A to B 1550 nm service. Find the root cause."


def iter_agent(case: Case, client, model: str, workdir: str, prompt: str = BASE_PROMPT,
               max_steps: int = 30, images: bool = True, temperature: Optional[float] = None,
               mode: str = "standard", vision_client=None, ticket: str = TICKET, remote_only: bool = False,
               verify: bool = True, reviewer=None, route_tool: bool = False):
    """Run the agent and yield events as it works (used by the demo app).

    Events: {"type": "thinking"}, {"type": "tool"}, and a final {"type": "done"}
    carrying the answer, grade, grounding check, checklist and usage.

    verify: run the runtime gate (gate.py) inside submit_diagnosis; rejected
    submissions go back to the agent, and after MAX_ATTEMPTS the ticket is
    escalated. reviewer: optional (client, model) for the second-model review.
    """
    tb = ToolBox(case, workdir, mode, vision_client=vision_client, remote_only=remote_only, route_tool=route_tool)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": ticket}]
    transcript, answer, usage = [], None, {"prompt_tokens": 0, "completion_tokens": 0}
    attempts, escalated, checks = 0, False, []
    t0 = time.time()
    for step in range(max_steps):
        kw = {"model": model, "messages": messages, "tools": _tool_list(tb), "tool_choice": "auto"}
        if temperature is not None:
            kw["temperature"] = temperature
        resp = client.chat.completions.create(**kw)
        if getattr(resp, "usage", None):
            usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage["completion_tokens"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        entry = {"step": step, "content": msg.content, "reasoning": reasoning, "tool_calls": []}
        if reasoning or msg.content:
            yield {"type": "thinking", "step": step, "reasoning": reasoning, "content": msg.content}
        calls = msg.tool_calls or []
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in calls]} if calls else {"role": "assistant", "content": msg.content or ""})
        if not calls:
            transcript.append(entry)
            messages.append({"role": "user", "content": "Continue the investigation with the tools, then call submit_diagnosis."})
            continue
        pending_images = []
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            image_path = None
            if name == "submit_diagnosis" and not verify:
                answer = args
                result = {"status": "submitted"}
            elif name == "submit_diagnosis":
                problems = deterministic_checks(args, tb)
                rev = None
                if not problems and reviewer is not None:
                    rev = review(reviewer[0], reviewer[1], args, tb)
                    if rev["verdict"] == "reject":
                        problems = ["reviewer: " + r for r in rev["reasons"]] or ["reviewer rejected the diagnosis"]
                attempts += 1
                checks.append({"attempt": attempts, "answer": args, "problems": problems, "review": rev})
                yield {"type": "verification", "step": step, "attempt": attempts, "max_attempts": MAX_ATTEMPTS,
                       "ok": not problems, "problems": problems, "review": rev, "answer": args}
                if not problems:
                    answer = args
                    result = {"status": "accepted", "checks": "all passed"}
                elif attempts >= MAX_ATTEMPTS:
                    answer, escalated = args, True
                    result = {"status": "escalated", "problems": problems,
                              "note": "Still failing after 3 attempts: the ticket is escalated to a human engineer."}
                else:
                    result = {"status": "rejected", "attempt": attempts, "max_attempts": MAX_ATTEMPTS,
                              "problems": problems, "note": "Fix these problems, then call submit_diagnosis again."}
            elif args is None:
                result = {"error": "arguments were not valid JSON"}
            else:
                try:
                    result = tb.call(name, args)
                except ToolError as e:
                    result = {"error": str(e)}
            if isinstance(result, dict) and "image_path" in result:
                image_path = result["image_path"]
                if images:
                    pending_images.append((image_path, name))
                result = {k: v for k, v in result.items() if k != "image_path"}
                result["image"] = "attached in the next message" if images else "not available"
            entry["tool_calls"].append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, default=str)})
            if name != "submit_diagnosis":
                yield {"type": "tool", "step": step, "name": name, "args": args, "result": result,
                       "image_path": image_path, "checklist": [dict(i) for i in tb.checklist]}
        for path, name in pending_images:
            messages.append(_image_message(path, name))
        transcript.append(entry)
        if answer is not None:
            break
    yield {"type": "done", "seed": case.seed, "model": model, "mode": mode, "answer": answer,
           "view": "noc" if remote_only else "field",
           "grade": grade(answer, case), "dispatch": grade_dispatch(answer, case),
           "grounding": grounding(answer, tb.calls), "checklist": tb.checklist,
           "verification": {"enabled": verify, "attempts": attempts, "escalated": escalated, "history": checks},
           "steps": len(transcript), "tool_calls": len(tb.calls), "seconds": round(time.time() - t0, 1),
           "usage": usage, "transcript": transcript}


def run_agent(case: Case, client, model: str, workdir: str, prompt: str = BASE_PROMPT,
              max_steps: int = 30, images: bool = True, temperature: Optional[float] = None,
              log_path: Optional[str] = None, mode: str = "standard", vision_client=None,
              remote_only: bool = False, verify: bool = True, reviewer=None, route_tool: bool = False) -> dict:
    result = None
    ticket = NOC_TICKET if remote_only else TICKET
    for ev in iter_agent(case, client, model, workdir, prompt, max_steps, images, temperature, mode, vision_client,
                         ticket, remote_only, verify, reviewer, route_tool):
        if ev["type"] == "done":
            result = {k: v for k, v in ev.items() if k != "type"}
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return result


def _load_env(path: str = ".env") -> None:
    """Read KEY=value lines from a local .env file (never committed)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client(provider: str = "baseten", model: Optional[str] = None):
    from openai import OpenAI
    _load_env()
    if provider == "baseten":
        return OpenAI(base_url="https://inference.baseten.co/v1", api_key=os.environ["BASETEN_API_KEY"]), \
            model or "meta-models/muse-spark-open-1026"
    if provider == "meta":
        return OpenAI(base_url=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"),
                      api_key=os.environ["META_API_KEY"]), (model or os.environ.get("META_MODEL", "muse-spark-1.3"))
    if provider == "glimmer":  # Muse Glimmer 30B served with vLLM on Modal (deploy/modal_glimmer.py)
        return OpenAI(base_url=os.environ["GLIMMER_BASE_URL"], api_key=os.environ["GLIMMER_API_KEY"],
                      timeout=1800), (model or "muse-glimmer")
    raise ValueError(provider)

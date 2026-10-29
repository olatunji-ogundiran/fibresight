"""FibreSight demo app.

  python -m fibresight.app            # then open http://localhost:8000

NOC console: a queue of alarmed links. Pick one and FibreSight diagnoses it
remotely (alarms, performance history, the embedded in-service OTDR), then
writes the dispatch work order: who to send, where exactly, with what kit.
The investigation view shows the agent's checklist, each instrument it reads,
its cited evidence and the diagnosis, checked against the simulator's truth.
Every run is saved and can be replayed offline.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import BASE_PROMPT, NOC_PROMPT, NOC_TICKET, TICKET, iter_agent, make_client
from .sim import FAULT_TYPES, generate_case
from .tools import ToolBox
from .workorder import work_order
from .field import FieldSession

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(os.getcwd(), "results", "demo"))
os.makedirs(WORK, exist_ok=True)
DEMO_RUNS = os.path.join(os.path.dirname(ROOT), "demo_runs")


def _seed_demo_runs() -> None:
    """Copy the curated demo runs shipped with the repo into the run folder, so they can be
    replayed on any machine without API keys."""
    import shutil
    if not os.path.isdir(DEMO_RUNS):
        return
    for name in os.listdir(DEMO_RUNS):
        src, dst = os.path.join(DEMO_RUNS, name), os.path.join(WORK, name)
        if os.path.exists(dst):
            continue
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)


_seed_demo_runs()

MODELS = {
    "spark-open": ("baseten", None, "Muse Spark Open (open weights, Baseten)"),
    "spark-1.3": ("meta", "muse-spark-1.3", "Muse Spark 1.3 (Meta Model API)"),
    "wonder": ("meta", "muse-wonder-1.0", "Muse Wonder 1.0 (Meta Model API)"),
    "glimmer": ("glimmer", None, "Muse Glimmer 30B (open weights, own GPU on Modal)"),
}

app = FastAPI(title="FibreSight")
app.mount("/work", StaticFiles(directory=WORK), name="work")


def _url(path):
    if not path:
        return None
    rel = os.path.relpath(os.path.abspath(path), WORK)
    return "/work/" + rel.replace(os.sep, "/")


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, default=str) + "\n\n"


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/api/models")
def models():
    from .agent import _load_env
    _load_env()
    return [{"id": k, "label": v[2]} for k, v in MODELS.items()
            if v[0] != "glimmer" or os.environ.get("GLIMMER_BASE_URL")]


SITES = ["Harbour POP", "Central DC", "North Exchange", "Airport Hub", "Riverside POP", "Hillcrest Exchange",
         "East Gate POP", "Lakeside Exchange", "Industrial Park POP", "University Hub", "West Ridge Exchange",
         "Market Street POP", "Old Town Exchange", "Port Road Hub"]


def site_names(seed: int) -> tuple:
    a = SITES[(seed * 5) % len(SITES)]
    b = SITES[(seed * 5 + 3 + seed % 4) % len(SITES)]
    return a, b


SEVERITY = {"critical": 0, "major": 1, "minor": 2}


def _ticket(seed: int, mode: str) -> dict:
    case = generate_case(seed, hard=(mode == "hard"))
    tb = ToolBox(case, os.path.join(WORK, "_preview"), mode, remote_only=True)
    alarms = tb.get_alarms()["alarms"]
    active = [a for a in alarms if a["cleared"] == "active"]
    flapping = [a for a in alarms if a["cleared"] != "active" and a["alarm"].startswith("RX_LOW")]
    shown = sorted(active, key=lambda a: SEVERITY.get(a["severity"], 3)) or sorted(flapping, key=lambda a: a["raised"])[-1:]
    top = shown[0] if shown else {"alarm": "CUSTOMER_COMPLAINT", "severity": "minor", "raised": "", "cleared": ""}
    a, b = site_names(seed)
    return {"seed": seed, "ticket": f"FS-{seed}", "link": f"{a} to {b}", "site_a": a, "site_b": b,
            "route_km": round(case.route_km, 2), "alarm": top["alarm"], "severity": top["severity"],
            "since": top["raised"], "state": "active" if active else f"raises and clears ({len(flapping)} times)",
            "service": "down" if top["alarm"] == "RX_LOS" else "degraded"}


@app.get("/api/queue")
def queue(mode: str = "standard", start: int = 1000, n: int = 8):
    items = [_ticket(s, mode) for s in range(start, start + min(max(n, 1), 20))]
    return sorted(items, key=lambda t: (SEVERITY.get(t["severity"], 3), t["since"]))


@app.get("/api/case/{seed}")
def case_info(seed: int, mode: str = "standard"):
    """Public view of a case: what a NOC engineer sees before investigating."""
    case = generate_case(seed, hard=(mode == "hard"))
    tb = ToolBox(case, os.path.join(WORK, "_preview"), mode)
    rec = tb.get_link_record()
    alarms = tb.get_alarms()["alarms"]
    site_a, site_b = site_names(seed)
    return {"seed": seed, "mode": mode, "route_km": case.route_km, "site_a": site_a, "site_b": site_b,
            "elements": rec["elements_in_route_order"], "alarms": alarms[-6:],
            "notes": tb.get_site_notes()["notes"], "service": rec["service"]}


REVIEWERS = {"spark-1.3": ("muse-spark-1.3", "Muse Spark 1.3"), "wonder": ("muse-wonder-1.0", "Muse Wonder 1.0")}


@app.get("/api/run")
def run(seed: int, mode: str = "standard", model: str = "spark-open", sam: bool = True, view: str = "noc",
        reviewer: str = "wonder"):
    if model not in MODELS:
        raise HTTPException(400, "unknown model")
    if view not in ("noc", "field"):
        raise HTTPException(400, "view must be noc or field")
    noc = view == "noc"
    provider, model_id, label = MODELS[model]
    run_id = f"{seed}_{mode}_{view}_{model}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
    workdir = os.path.join(WORK, run_id)
    case = generate_case(seed, hard=(mode == "hard"))
    client, mid = make_client(provider, model_id)
    vision = make_client("meta")[0] if (sam and not noc) else None
    rev = (make_client("meta", REVIEWERS[reviewer][0])[0], REVIEWERS[reviewer][0]) if reviewer in REVIEWERS else None
    rev_label = REVIEWERS[reviewer][1] if rev else "off"

    def stream():
        log = open(os.path.join(WORK, run_id + ".jsonl"), "w")

        def emit(ev):
            log.write(json.dumps(ev, default=str) + "\n")
            log.flush()
            return _sse(ev)

        yield emit({"type": "start", "run_id": run_id, "seed": seed, "mode": mode, "view": view, "model": label,
                    "reviewer": rev_label})
        try:
            for ev in iter_agent(case, client, mid, workdir, NOC_PROMPT if noc else BASE_PROMPT, mode=mode,
                                 vision_client=vision, ticket=NOC_TICKET if noc else TICKET, remote_only=noc,
                                 verify=True, reviewer=rev, route_tool=True):
                if ev["type"] == "tool":
                    ev["image_url"] = _url(ev.pop("image_path"))
                if ev["type"] == "done":
                    ev.pop("transcript", None)
                    ev["truth"] = case.truth
                    rec = ToolBox(case, os.path.join(WORK, "_preview"), mode).get_link_record()
                    ev["work_order"] = work_order(ev.get("answer"), rec, f"FS-{seed}")
                    ev["reviewer"] = rev_label
                yield emit(ev)
        except Exception as e:  # surface provider errors in the UI
            yield emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        log.close()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/signoff")
def signoff(payload: dict):
    """The NOC engineer approves the work order before it is dispatched (human in the loop)."""
    run_id = os.path.basename(str(payload.get("run_id", "")))
    path = os.path.join(WORK, run_id + ".jsonl")
    ev = {"type": "signoff", "run_id": run_id, "by": str(payload.get("by") or "NOC engineer")[:60],
          "decision": "approved" if payload.get("approved", True) else "returned",
          "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(path):
        with open(path, "a") as f:
            f.write(json.dumps(ev) + "\n")
    return ev


@app.get("/api/runs")
def runs():
    files = sorted((f for f in os.listdir(WORK) if f.endswith(".jsonl") and not f.startswith("field_")), reverse=True)
    out = []
    for f in files[:40]:
        try:
            done = [json.loads(l) for l in open(os.path.join(WORK, f)) if '"type": "done"' in l]
            d = done[-1] if done else None
            score = (d.get("dispatch") if d.get("view") == "noc" else d["grade"])["score"] if d else None
        except Exception:
            score = None
        out.append({"run_id": f[:-6], "score": score})
    return out


@app.get("/api/replay/{run_id}")
def replay(run_id: str, speed: float = 1.0):
    path = os.path.join(WORK, os.path.basename(run_id) + ".jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "run not found")

    def stream():
        for line in open(path):
            ev = json.loads(line)
            ev["replay"] = True
            yield _sse(ev)
            time.sleep({"thinking": 1.2, "tool": 1.6}.get(ev["type"], 0.4) / max(speed, 0.1))

    return StreamingResponse(stream(), media_type="text/event-stream")


# ------------------------------------------------------------------ field assistant
SESSIONS: dict = {}


def _field_events(events: list) -> list:
    out = []
    for ev in events:
        ev = dict(ev)
        for k in ("image_path", "sam_image_path"):
            if ev.get(k):
                ev[k.replace("_path", "_url")] = _url(ev.pop(k))
        out.append(ev)
    return out


def _field_log(fs, events: list, meta: dict = None) -> None:
    """Append a field session's events to results/demo/<session>.jsonl so it can be replayed."""
    path = os.path.join(WORK, os.path.basename(fs.workdir) + ".jsonl")
    with open(path, "a") as f:
        if meta:
            f.write(json.dumps(dict(meta, type="field_start"), default=str) + "\n")
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")


@app.get("/api/field/sessions")
def field_sessions():
    out = []
    for name in sorted((n for n in os.listdir(WORK) if n.startswith("field_") and n.endswith(".jsonl")), reverse=True)[:40]:
        lines = [json.loads(l) for l in open(os.path.join(WORK, name))]
        closed = any(e.get("type") == "close" and e.get("ok") for e in lines)
        seed = lines[0].get("seed") if lines else None
        out.append({"id": name[:-6], "seed": seed, "closed": closed, "turns": len(lines) - 1})
    return out


@app.get("/api/field/session/{name}")
def field_session(name: str):
    path = os.path.join(WORK, os.path.basename(name) + ".jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "session not found")
    lines = [json.loads(l) for l in open(path)]
    return {"meta": lines[0] if lines else {}, "events": lines[1:]}


@app.post("/api/field/start")
def field_start(payload: dict):
    """Open a technician session for a ticket, with the NOC's diagnosis if there is one."""
    seed, mode = int(payload.get("seed", 1000)), payload.get("mode", "standard")
    model = payload.get("model", "spark-open")
    if model not in MODELS:
        raise HTTPException(400, "unknown model")
    provider, model_id, label = MODELS[model]
    client, mid = make_client(provider, model_id)
    vision = make_client("meta")[0] if payload.get("sam", True) else None
    case = generate_case(seed, hard=(mode == "hard"))
    sid_dir = os.path.join(WORK, f"field_{seed}_{uuid.uuid4().hex[:6]}")
    fs = FieldSession(case, sid_dir, client, mid, vision, payload.get("answer"), f"FS-{seed}")
    SESSIONS[fs.id] = fs
    try:
        events = fs.start()
    except Exception as e:
        events = [{"type": "error", "message": f"{type(e).__name__}: {e}"}]
    out = _field_events(events)
    _field_log(fs, out, {"seed": seed, "mode": mode, "session": fs.id, "work_order": fs.work_order, "model": label})
    return {"session": fs.id, "work_order": fs.work_order, "model": label, "events": out}


@app.post("/api/field/{sid}/say")
async def field_say(sid: str, text: str = Form(None), instrument: str = Form(None), work_done: str = Form(None),
                    photo_label: str = Form(None), photo: UploadFile = File(None), voice: UploadFile = File(None)):
    fs = SESSIONS.get(sid)
    if fs is None:
        raise HTTPException(404, "session not found (the server may have restarted)")
    photo_path, is_voice, pre = None, False, []
    if voice is not None:
        from .voice import FIBRE_KEYWORDS, transcribe_file
        path = os.path.join(fs.workdir, f"voice_{uuid.uuid4().hex[:6]}{os.path.splitext(voice.filename or '')[1] or '.webm'}")
        open(path, "wb").write(await voice.read())
        try:
            heard = (await asyncio.to_thread(transcribe_file, path, None, FIBRE_KEYWORDS))["text"]
        except Exception as e:
            return {"events": [{"type": "error", "message": f"Transcription failed: {e}"}]}
        if not heard.strip():
            return {"events": [{"type": "error", "message": "No speech heard in the voice note. Try again closer to the microphone."}]}
        text, is_voice = ((text + " ") if text else "") + heard, True
    if photo is not None:
        photo_path = os.path.join(fs.workdir, f"upload_{uuid.uuid4().hex[:6]}{os.path.splitext(photo.filename or '')[1] or '.jpg'}")
        open(photo_path, "wb").write(await photo.read())
    try:
        events = await asyncio.to_thread(fs.technician, text=text, photo_path=photo_path, photo_label=photo_label,
                                         instrument=json.loads(instrument) if instrument else None,
                                         work_done=json.loads(work_done) if work_done else None, voice=is_voice)
    except Exception as e:
        events = [{"type": "error", "message": f"{type(e).__name__}: {e}"}]
    out = _field_events(pre + events)
    _field_log(fs, out)
    return {"events": out, "closed": fs.closed}


@app.get("/api/fault_types")
def fault_types():
    return FAULT_TYPES


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

"""FibreSight demo app.

  python -m fibresight.app            # then open http://localhost:8000

Pick a case, press Run, and watch the agent investigate: its checklist, each
instrument it uses (OTDR traces, end-face images with SAM 3.1 boxes, power
readings), its cited evidence and the final diagnosis, checked against the
simulator's ground truth. Every run is saved and can be replayed offline.
"""
from __future__ import annotations

import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import BASE_PROMPT, iter_agent, make_client
from .sim import FAULT_TYPES, generate_case
from .tools import ToolBox

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(os.getcwd(), "results", "demo"))
os.makedirs(WORK, exist_ok=True)

MODELS = {
    "spark-open": ("baseten", None, "Muse Spark Open (open weights, Baseten)"),
    "spark-1.3": ("meta", "muse-spark-1.3", "Muse Spark 1.3 (Meta Model API)"),
    "wonder": ("meta", "muse-wonder-1.0", "Muse Wonder 1.0 (Meta Model API)"),
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
    return [{"id": k, "label": v[2]} for k, v in MODELS.items()]


@app.get("/api/case/{seed}")
def case_info(seed: int, mode: str = "standard"):
    """Public view of a case: what a NOC engineer sees before investigating."""
    case = generate_case(seed, hard=(mode == "hard"))
    tb = ToolBox(case, os.path.join(WORK, "_preview"), mode)
    rec = tb.get_link_record()
    alarms = tb.get_alarms()["alarms"]
    return {"seed": seed, "mode": mode, "route_km": case.route_km,
            "elements": rec["elements_in_route_order"], "alarms": alarms[-6:],
            "notes": tb.get_site_notes()["notes"], "service": rec["service"]}


@app.get("/api/run")
def run(seed: int, mode: str = "standard", model: str = "spark-open", sam: bool = True):
    if model not in MODELS:
        raise HTTPException(400, "unknown model")
    provider, model_id, label = MODELS[model]
    run_id = f"{seed}_{mode}_{model}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
    workdir = os.path.join(WORK, run_id)
    case = generate_case(seed, hard=(mode == "hard"))
    client, mid = make_client(provider, model_id)
    vision = make_client("meta")[0] if sam else None

    def stream():
        log = open(os.path.join(WORK, run_id + ".jsonl"), "w")

        def emit(ev):
            log.write(json.dumps(ev, default=str) + "\n")
            log.flush()
            return _sse(ev)

        yield emit({"type": "start", "run_id": run_id, "seed": seed, "mode": mode, "model": label})
        try:
            for ev in iter_agent(case, client, mid, workdir, BASE_PROMPT, mode=mode, vision_client=vision):
                if ev["type"] == "tool":
                    ev["image_url"] = _url(ev.pop("image_path"))
                if ev["type"] == "done":
                    ev.pop("transcript", None)
                    ev["truth"] = case.truth
                yield emit(ev)
        except Exception as e:  # surface provider errors in the UI
            yield emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        log.close()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs")
def runs():
    files = sorted((f for f in os.listdir(WORK) if f.endswith(".jsonl")), reverse=True)
    out = []
    for f in files[:40]:
        try:
            done = [json.loads(l) for l in open(os.path.join(WORK, f)) if '"type": "done"' in l]
            score = done[-1]["grade"]["score"] if done else None
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


@app.get("/api/fault_types")
def fault_types():
    return FAULT_TYPES


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

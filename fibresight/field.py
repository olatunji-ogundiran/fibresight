"""Field assistant: a chat between FibreSight and the technician on site.

The NOC's work order says where to go and what to bring. On site the
technician talks (voice notes, transcribed by Muse Voice Transcribe), sends
end-face pictures (checked by SAM 3.1) and instrument readings, and reports
work done. The assistant guides one step at a time and can only close the
ticket when a live reading after the repair is back in range, so a fix is
proven by measurement, not by someone's word.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid

from PIL import Image

from .sim import RX_HIGH_WARN, RX_LOW_WARN, Case
from .tools import ToolBox, ToolError
from .workorder import work_order

FIELD_PROMPT = """You are FibreSight's field assistant, talking to a fibre technician on site through a phone app.
The NOC dispatched them with the work order below. Guide them one step at a time: what to do, where, with which
instrument, and what reading or picture to send back. Keep each message short and plain, because they are working
with their hands: at most 5 short lines, numbered when there is more than one step. Never invent a reading.
Readings and pictures come from the technician's instruments; trust measurements over guesses, and if a reading
contradicts the work order, say so and adjust the plan.
End-face pictures are analysed by Meta SAM 3.1: you get the picture and its defect report. Rules of the trade:
inspect before you connect; a dirty end face is cleaned and inspected again; a scratch or chip in the core zone
cannot be cleaned, so the connector or patch cord is replaced. After any repair, take a power reading and compare it
with the design range (receiver about -10 to -12.5 dBm; alarm below {low:.0f} dBm).
You can read the live receiver power from the network yourself with read_live_rx at any time: do it when the
technician reports a repair. When a reading taken after the repair shows the service back in range, call close_ticket.
Never close a ticket on the technician's word alone.

Work order:
{work_order}

Link (as-built route, route km from site A):
{route}"""

CLOSE_SPEC = {"type": "function", "function": {
    "name": "close_ticket",
    "description": "Close the ticket once a reading taken after the repair proves the service is back in range.",
    "parameters": {"type": "object", "properties": {
        "fault_confirmed": {"type": "string", "description": "What was actually found on site"},
        "work_done": {"type": "string"},
        "reading_before": {"type": "string"}, "reading_after": {"type": "string"},
        "summary": {"type": "string", "description": "One or two lines for the NOC"}},
        "required": ["fault_confirmed", "work_done", "reading_after", "summary"]}}}

LIVE_RX_SPEC = {"type": "function", "function": {
    "name": "read_live_rx",
    "description": "Current receiver power of the service at site B (B-RX), read live from the network management system. Use it before and after repairs; the technician does not need to be at site B.",
    "parameters": {"type": "object", "properties": {}}}}

INSTRUMENTS = {
    "power": "Power meter",
    "scope": "Inspection scope",
    "otdr": "Handheld OTDR",
}


def _png(path: str, out: str, max_side: int = 640) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(out)
    return out


def _image_part(path: str) -> dict:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class FieldSession:
    def __init__(self, case: Case, workdir: str, client, model: str, vision_client=None,
                 answer: dict = None, ticket: str = None):
        self.id = uuid.uuid4().hex[:10]
        self.case, self.client, self.model = case, client, model
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.tb = ToolBox(case, workdir, "standard", vision_client=vision_client)
        self.vision = vision_client
        record = self.tb.get_link_record()
        self.work_order = work_order(answer, record, ticket or f"FS-{case.seed}") if answer else {}
        route = "; ".join(f"{e['id']} {e['route_km']:.3f}" for e in record["elements_in_route_order"])
        wo = json.dumps({k: v for k, v in self.work_order.items() if k != "evidence"}, indent=1) if self.work_order \
            else "No NOC diagnosis yet: start from the alarms and measurements on site."
        self.messages = [{"role": "system", "content": FIELD_PROMPT.format(work_order=wo, route=route, low=RX_LOW_WARN)}]
        self.closed = None
        self.last_fix_at = None   # index into readings when the last repair was reported
        self.readings = []        # (time, point, dBm)
        self.log = []

    # ---------------------------------------------------------------- inputs
    def technician(self, text: str = None, photo_path: str = None, photo_label: str = None,
                   instrument: dict = None, work_done: dict = None, voice: bool = False) -> list:
        """One technician turn. Returns events for the UI."""
        events, parts = [], []
        if text:
            parts.append({"type": "text", "text": ("Technician (voice note): " if voice else "Technician: ") + text})
            events.append({"type": "tech_text", "text": text, "voice": voice})
        if work_done:
            r = self.tb.apply_fix(work_done.get("action"), work_done.get("element_id"), work_done.get("route_km"))
            self.last_fix_at = len(self.readings)
            where = r["element_id"] or (f"route km {r['route_km']}" if r["route_km"] is not None else "")
            msg = f"[Work done] {r['work_done'].replace('_', ' ')} {('at ' + where) if where else ''}".strip()
            parts.append({"type": "text", "text": msg})
            events.append({"type": "work_done", "text": msg})
        if instrument:
            ev, part, img = self._instrument(instrument)
            events.append(ev)
            parts.append({"type": "text", "text": part})
            if img:
                parts.append(_image_part(img))
        if photo_path:
            ev, part, imgs = self._photo(photo_path, photo_label)
            events.append(ev)
            parts.append({"type": "text", "text": part})
            parts += [_image_part(p) for p in imgs]
        if not parts:
            return [{"type": "error", "message": "nothing to send"}]
        self.messages.append({"role": "user", "content": parts})
        events += self._reply()
        self.log += events
        return events

    def start(self) -> list:
        self.messages.append({"role": "user", "content": "Technician: I have the work order and I'm on my way. What's the plan?"})
        events = [{"type": "tech_text", "text": "I have the work order and I'm on my way. What's the plan?", "voice": False}]
        events += self._reply()
        self.log += events
        return events

    # ---------------------------------------------------------------- tools
    def _instrument(self, ins: dict):
        kind = ins.get("kind")
        try:
            if kind == "power":
                r = self.tb.call("measure_power", {"point": ins.get("point", "B-RX")})
                self.readings.append((time.time(), r["point"], r["power_dbm"]))
                val = r["reading"] if r["power_dbm"] is None else f"{r['power_dbm']:.2f} dBm"
                text = f"[Power meter, 1550 nm, at {r['point']}] {val}"
                return {"type": "reading", "instrument": "power", "point": r["point"], "value": r["power_dbm"],
                        "text": text}, text, None
            if kind == "otdr":
                wl, site = int(ins.get("wavelength_nm", 1550)), ins.get("from_site", "A")
                r = self.tb.call("otdr_event_table", {"wavelength_nm": wl, "from_site": site})
                rows = [{k: e[k] for k in ("event", "distance_km", "type", "loss_db", "reflectance_db")} for e in r["events"]]
                text = f"[Handheld OTDR, {wl} nm, from site {site}, fibre km] " + json.dumps(rows)
                return {"type": "reading", "instrument": "otdr", "wavelength_nm": wl, "from_site": site,
                        "events": rows, "text": f"OTDR {wl} nm from site {site}: {len(rows)} events"}, text, None
            if kind == "scope":
                eid = ins.get("element_id", "B-ODF")
                r = self.tb.call("inspect_connector", {"element_id": eid})
                ev, part, imgs = self._photo(r["image_path"], eid)
                return ev, part, (imgs[-1] if imgs else None)
        except ToolError as e:
            return {"type": "error", "message": str(e)}, f"[Instrument error] {e}", None
        return {"type": "error", "message": "unknown instrument"}, "[Instrument error] unknown instrument", None

    def _photo(self, path: str, label: str = None):
        tag = uuid.uuid4().hex[:6]
        png = _png(path, os.path.join(self.workdir, f"photo_{tag}.png"))
        ev = {"type": "photo", "label": label, "image_path": png}
        if self.vision is None:
            return ev, f"[End-face picture{' of ' + label if label else ''}] (SAM 3.1 off)", [png]
        from .vision import segment_endface
        try:
            seg = segment_endface(self.vision, png, os.path.join(self.workdir, f"sam_{tag}.png"))
        except Exception as e:  # keep the chat going if the vision call fails
            return dict(ev, sam_error=str(e)), f"[End-face picture{' of ' + label if label else ''}] SAM 3.1 failed: {e}", [png]
        ev.update(sam_image_path=seg["annotated_image_path"], verdict=seg["verdict"],
                  particles=seg["particles_in_zone"], scratches=seg["scratches_in_zone"])
        report = (f"[End-face picture{' of ' + label if label else ''}, SAM 3.1 report] verdict: {seg['verdict']}; "
                  f"particles in zone: {seg['particles_in_zone']}; scratches in zone: {seg['scratches_in_zone']}; "
                  f"outside zone: {seg['outside_zone']}. Annotated picture attached (red = scratch, yellow = particle).")
        return ev, report, [seg["annotated_image_path"]]

    def _close(self, args: dict) -> dict:
        """Check the close request against the live reading, not the technician's word."""
        after = [r for r in self.readings[self.last_fix_at or 0:] if r[1] == "B-RX"] if self.last_fix_at is not None else []
        if not after:
            return {"error": "Cannot close: no receiver (B-RX) reading has been taken since the repair was reported."}
        val = after[-1][2]
        if val is None or not (RX_LOW_WARN < val < RX_HIGH_WARN):
            return {"error": f"Cannot close: the latest B-RX reading after the repair is {val} dBm, outside the range "
                             f"{RX_LOW_WARN:.0f} to {RX_HIGH_WARN:.0f} dBm."}
        self.closed = dict(args, reading_checked=f"B-RX {val:.2f} dBm", fault_removed=bool(getattr(self.tb.case, "repaired", False)))
        return {"status": "closed", "reading_checked": f"B-RX {val:.2f} dBm"}

    # ---------------------------------------------------------------- model
    def _reply(self, max_rounds: int = 4) -> list:
        events = []
        for _ in range(max_rounds):
            resp = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                       tools=[LIVE_RX_SPEC, CLOSE_SPEC], tool_choice="auto")
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            self.messages.append({"role": "assistant", "content": msg.content or "",
                                  **({"tool_calls": [{"id": c.id, "type": "function", "function": {
                                      "name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {})})
            if msg.content and msg.content.strip():
                events.append({"type": "assistant", "text": msg.content.strip()})
            if not calls:
                break
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if c.function.name == "read_live_rx":
                    r = self.tb.call("measure_power", {"point": "B-RX"})
                    self.readings.append((time.time(), "B-RX", r["power_dbm"]))
                    val = "no light (loss of signal)" if r["power_dbm"] is None else f"{r['power_dbm']:.2f} dBm"
                    out = {"B-RX_live_power": val, "source": "network management system, live PM"}
                    events.append({"type": "reading", "instrument": "nms", "point": "B-RX", "value": r["power_dbm"],
                                   "text": f"Live Rx from the network (B-RX): {val}"})
                elif c.function.name == "close_ticket":
                    out = self._close(args)
                    events.append({"type": "close", "ok": "error" not in out, "result": out,
                                   "report": self.closed if "error" not in out else None})
                else:
                    out = {"error": "unknown tool"}
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out)})
        return events

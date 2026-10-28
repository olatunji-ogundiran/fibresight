"""Exercise the agent loop offline with a scripted fake model."""
import json
from types import SimpleNamespace as NS

from fibresight.agent import run_agent
from fibresight.sim import generate_case


class ScriptedClient:
    """Plays a fixed sequence of tool calls, then submits the given answer."""

    def __init__(self, calls, answer):
        self.script = [(n, a) for n, a in calls] + [("submit_diagnosis", answer)]
        self.i = 0
        self.seen_images = 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.seen_images += sum(1 for m in kw["messages"] if m["role"] == "user" and isinstance(m["content"], list))
        name, args = self.script[self.i]
        self.i += 1
        call = NS(id=f"c{self.i}", function=NS(name=name, arguments=json.dumps(args)))
        msg = NS(content=None, tool_calls=[call], reasoning_content="thinking")
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def test_agent_loop_scores_correct_answer(tmp_path):
    case = generate_case(1)  # damaged connector
    eid = case.truth["element_id"]
    client = ScriptedClient(
        [("get_link_record", {}), ("get_alarms", {}), ("otdr_event_table", {"wavelength_nm": 1550, "from_site": "A"}),
         ("inspect_connector", {"element_id": eid}), ("bogus_tool", {})],
        {"fault_type": "damaged_connector", "element_id": eid, "action": "replace_connector", "evidence": "scratches"})
    out = run_agent(case, client, "fake", str(tmp_path), log_path=str(tmp_path / "log.json"), verify=False)
    assert out["grade"]["score"] == 1.0
    assert client.seen_images >= 1
    assert out["transcript"][4]["tool_calls"][0]["result"]["error"].startswith("unknown tool")
    assert (tmp_path / "log.json").exists()


def test_agent_loop_wrong_answer_scores_zero(tmp_path):
    case = generate_case(4)  # fibre break
    client = ScriptedClient([("get_alarms", {})],
                            {"fault_type": "fibre_break", "route_km": 1.0, "action": "repair_break", "evidence": "guess"})
    assert run_agent(case, client, "fake", str(tmp_path), verify=False)["grade"]["score"] == 0.0


def test_no_submission_scores_zero(tmp_path):
    case = generate_case(2)
    client = ScriptedClient([("get_alarms", {})] * 40, {})
    out = run_agent(case, client, "fake", str(tmp_path), max_steps=5, verify=False)
    assert out["answer"] is None and out["grade"]["score"] == 0.0


# ---------------------------------------------------------------- field assistant
from fibresight.field import FieldSession


class FieldClient:
    """Each create() pops the next scripted turn: a text reply or a list of tool calls."""

    def __init__(self, turns):
        self.turns, self.i = list(turns), 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        t = self.turns[self.i] if self.i < len(self.turns) else "ok"
        self.i += 1
        if isinstance(t, str):
            msg = NS(content=t, tool_calls=None)
        else:
            msg = NS(content=None, tool_calls=[NS(id=f"f{self.i}_{k}", function=NS(name=n, arguments=json.dumps(a)))
                                               for k, (n, a) in enumerate(t)])
        return NS(choices=[NS(message=msg)], usage=None)


CLOSE = ("close_ticket", {"fault_confirmed": "x", "work_done": "y", "reading_after": "z", "summary": "s"})


def test_field_close_requires_reading_after_a_real_fix(tmp_path):
    case = generate_case(1)  # damaged connector
    eid = case.truth["element_id"]
    answer = {"fault_type": "dirty_connector", "element_id": eid, "action": "clean_connector", "evidence": "remote"}
    client = FieldClient([
        "Go to the connector and send a picture.",           # start
        [CLOSE], "Need a reading first.",                    # tries to close on the tech's word: refused
        [("read_live_rx", {}), CLOSE], "Still low.",         # cleaned a scratched connector: reading still low, refused
        [("read_live_rx", {}), CLOSE], "Closed.",            # replaced it: reading in range, closed
    ])
    fs = FieldSession(case, str(tmp_path), client, "fake", None, answer)
    assert fs.work_order["send"] == "Patching technician"
    fs.start()
    ev = fs.technician(text="It's done, close it")
    assert any(e["type"] == "close" and not e["ok"] for e in ev) and fs.closed is None
    ev = fs.technician(work_done={"action": "clean_connector", "element_id": eid})
    assert any(e["type"] == "close" and not e["ok"] for e in ev) and fs.closed is None
    ev = fs.technician(work_done={"action": "replace_connector", "element_id": eid})
    assert any(e["type"] == "close" and e["ok"] for e in ev)
    assert fs.closed["fault_removed"] is True


# ---------------------------------------------------------------- runtime gate
from fibresight.gate import deterministic_checks
from fibresight.tools import ToolBox


class SeqClient(ScriptedClient):
    """Plays the given (tool, args) calls in order, submissions included."""

    def __init__(self, calls):
        super().__init__(calls[:-1], calls[-1][1])


def _good(case, refs=True):
    t = case.truth
    a = {"fault_type": t["fault_type"], "element_id": t["element_id"], "route_km": None,
         "action": t["action"], "evidence": "end face scratched"}
    if refs:
        a["evidence_refs"] = [{"call_id": 2, "quote": "splice closure"}, {"call_id": 2, "quote": "connector"}]
    return a


def test_gate_rejects_incomplete_then_accepts_fixed(tmp_path):
    case = generate_case(1)  # damaged connector
    client = SeqClient([("set_checklist", {"items": ["read record", "decide"]}), ("get_link_record", {}),
                        ("submit_diagnosis", _good(case, refs=False)),
                        ("update_checklist", {"index": 0, "status": "done"}),
                        ("update_checklist", {"index": 1, "status": "done"}),
                        ("submit_diagnosis", _good(case))])
    out = run_agent(case, client, "fake", str(tmp_path))
    v = out["verification"]
    assert v["attempts"] == 2 and not v["escalated"]
    first = " ".join(v["history"][0]["problems"])
    assert "evidence_refs" in first and "still open" in first
    assert v["history"][1]["problems"] == [] and out["grade"]["score"] == 1.0


def test_gate_escalates_after_three_rejections(tmp_path):
    case = generate_case(1)
    bad = _good(case, refs=False)
    client = SeqClient([("submit_diagnosis", bad)] * 3)
    out = run_agent(case, client, "fake", str(tmp_path))
    assert out["verification"]["attempts"] == 3 and out["verification"]["escalated"]


def test_gate_catches_bad_location_action_and_citation(tmp_path):
    case = generate_case(4)  # fibre break
    tb = ToolBox(case, str(tmp_path))
    tb.call("set_checklist", {"items": ["x"]}); tb.call("update_checklist", {"index": 0, "status": "done"})
    tb.call("get_alarms", {})
    ans = {"fault_type": "fibre_break", "route_km": 999.0, "action": "resplice", "evidence": "x",
           "element_id": "SC-99",
           "evidence_refs": [{"call_id": 3, "quote": "-123.456 dBm"}, {"call_id": 9, "quote": "x"}]}
    p = " | ".join(deterministic_checks(ans, tb))
    assert "SC-99" in p and "outside the route" in p and "does not fix" in p and "does not match" in p


class Reviewer:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        v = self.verdicts.pop(0)
        return NS(choices=[NS(message=NS(content=json.dumps({"verdict": v, "reasons": [f"{v} reason"]})))])


def test_reviewer_rejection_goes_back_to_agent(tmp_path):
    case = generate_case(1)
    good = _good(case)
    client = SeqClient([("set_checklist", {"items": ["a"]}), ("get_link_record", {}),
                        ("update_checklist", {"index": 0, "status": "done"}),
                        ("submit_diagnosis", good), ("submit_diagnosis", good)])
    out = run_agent(case, client, "fake", str(tmp_path), reviewer=(Reviewer(["reject", "approve"]), "judge"))
    h = out["verification"]["history"]
    assert h[0]["problems"] == ["reviewer: reject reason"] and h[1]["review"]["verdict"] == "approve"
    assert not out["verification"]["escalated"] and out["grade"]["score"] == 1.0

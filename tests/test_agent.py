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
    out = run_agent(case, client, "fake", str(tmp_path), log_path=str(tmp_path / "log.json"))
    assert out["grade"]["score"] == 1.0
    assert client.seen_images >= 1
    assert out["transcript"][4]["tool_calls"][0]["result"]["error"].startswith("unknown tool")
    assert (tmp_path / "log.json").exists()


def test_agent_loop_wrong_answer_scores_zero(tmp_path):
    case = generate_case(4)  # fibre break
    client = ScriptedClient([("get_alarms", {})],
                            {"fault_type": "fibre_break", "route_km": 1.0, "action": "repair_break", "evidence": "guess"})
    assert run_agent(case, client, "fake", str(tmp_path))["grade"]["score"] == 0.0


def test_no_submission_scores_zero(tmp_path):
    case = generate_case(2)
    client = ScriptedClient([("get_alarms", {})] * 40, {})
    out = run_agent(case, client, "fake", str(tmp_path), max_steps=5)
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

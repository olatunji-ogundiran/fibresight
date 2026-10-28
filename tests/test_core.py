import math

import pytest

from fibresight.expert import solve
from fibresight.sim import ACTION_FOR_FAULT, FAULT_TYPES, generate_case
from fibresight.tools import ToolBox, ToolError
from fibresight.verifier import grade, validate

SEEDS = range(0, 70)


def test_cases_are_deterministic():
    a, b = generate_case(5), generate_case(5)
    assert a.truth == b.truth and a.notes == b.notes


@pytest.mark.parametrize("seed", SEEDS)
def test_reference_expert_solves_every_case(seed, tmp_path):
    case = generate_case(seed)
    answer = solve(ToolBox(case, str(tmp_path)))
    assert grade(answer, case)["score"] == 1.0, (answer, case.truth)


def test_healthy_link_has_no_alarm_before_fault(tmp_path):
    for seed in range(40):
        case = generate_case(seed)
        alarms = ToolBox(case, str(tmp_path)).get_alarms()["alarms"]
        onset = case.onset.strftime("%Y-%m-%d %H:%M UTC")
        assert all(a["raised"] >= onset[:13] for a in alarms), (seed, alarms, onset)


@pytest.mark.parametrize("bad", [
    None, [], "dirty_connector", [{"fault_type": "dirty_connector"}],
    {"fault_type": "dirty", "action": "clean_connector", "evidence": "", "element_id": "A-ODF"},
    {"fault_type": "fibre_break", "action": "repair_break", "evidence": ""},                  # no route_km
    {"fault_type": "fibre_break", "action": "repair_break", "evidence": "", "route_km": math.nan},
    {"fault_type": "fibre_break", "action": "repair_break", "evidence": "", "route_km": True},
    {"fault_type": "bad_splice", "action": "resplice", "evidence": ""},                       # no element
    {"fault_type": "bad_splice", "action": "resplice", "evidence": "", "element_id": "SC-01", "guess2": 1},
])
def test_malformed_answers_score_zero(bad):
    case = generate_case(3)
    assert validate(bad)
    assert grade(bad, case)["score"] == 0.0


def test_constant_answers_cannot_pass_more_than_their_share():
    """Any fixed answer passes at most the cases of one fault type."""
    seeds = range(0, 140)
    cases = [generate_case(s) for s in seeds]
    consts = [
        {"fault_type": "tx_degradation", "element_id": "A-TX", "action": "replace_transmitter", "evidence": "x"},
        {"fault_type": "wrong_attenuator", "element_id": "B-ATT", "action": "replace_attenuator", "evidence": "x"},
        {"fault_type": "dirty_connector", "element_id": "B-ODF", "action": "clean_connector", "evidence": "x"},
        {"fault_type": "fibre_break", "route_km": 20.0, "action": "repair_break", "evidence": "x"},
        {"fault_type": "bad_splice", "element_id": "SC-02", "action": "resplice", "evidence": "x"},
    ]
    for c in consts:
        rate = sum(grade(c, k)["score"] for k in cases) / len(cases)
        assert rate <= 1.0 / len(FAULT_TYPES) + 0.01, (c, rate)


def test_wrong_action_or_location_fails(tmp_path):
    for seed in range(7):
        case = generate_case(seed)
        ans = solve(ToolBox(case, str(tmp_path)))
        wrong_action = dict(ans, action=next(a for a in set(ACTION_FOR_FAULT.values()) if a != ans["action"]))
        assert grade(wrong_action, case)["score"] == 0.0
        if ans.get("route_km") is not None and case.truth["fault_type"] in ("macrobend", "fibre_break"):
            assert grade(dict(ans, route_km=ans["route_km"] + 0.3), case)["score"] == 0.0


def test_tool_argument_errors(tmp_path):
    tb = ToolBox(generate_case(1), str(tmp_path))
    with pytest.raises(ToolError):
        tb.call("otdr_event_table", {"wavelength_nm": 1625, "from_site": "A"})
    with pytest.raises(ToolError):
        tb.call("measure_power", {"point": "C"})
    with pytest.raises(ToolError):
        tb.call("rm_rf", {})
    with pytest.raises(ToolError):
        tb.call("inspect_connector", {"element_id": "SC-01"})


def test_truth_not_exposed_by_tools(tmp_path):
    case = generate_case(9)
    tb = ToolBox(case, str(tmp_path))
    import json
    blob = json.dumps([tb.get_link_record(), tb.get_alarms(), tb.get_site_notes(),
                       tb.otdr_event_table(1550, "A"), tb.get_pm("rx_power_dbm")], default=str)
    for word in FAULT_TYPES:
        assert word not in blob


def test_grounding_detects_invented_readings(tmp_path):
    from fibresight.verifier import grounding
    tb = ToolBox(generate_case(2), str(tmp_path))
    rec = tb.call("measure_power", {"point": "A-TX"})
    real = str(rec["power_dbm"])
    good = {"evidence_refs": [{"call_id": rec["call_id"], "quote": f"A-TX {real} dBm"}]}
    fake = {"evidence_refs": [{"call_id": rec["call_id"], "quote": "A-TX -9.87 dBm"},
                              {"call_id": 99, "quote": "1.23 dB"}]}
    assert grounding(good, tb.calls)["rate"] == 1.0
    g = grounding(fake, tb.calls)
    assert g["rate"] == 0.0 and g["refs"] == 2
    assert grounding({"evidence_refs": []}, tb.calls)["rate"] is None


def test_checklist_tools(tmp_path):
    tb = ToolBox(generate_case(2), str(tmp_path))
    tb.call("set_checklist", {"items": ["alarms", "equipment"]})
    out = tb.call("update_checklist", {"index": 1, "status": "done", "note": "Tx normal"})
    assert out["checklist"][1]["status"] == "done"
    with pytest.raises(ToolError):
        tb.call("update_checklist", {"index": 5, "status": "done"})


# ---------------------------------------------------------------- NOC view
from fibresight.expert import solve_remote
from fibresight.verifier import grade_dispatch
from fibresight.tools import REMOTE_TOOLS


def test_noc_toolbox_has_remote_tools_only(tmp_path):
    tb = ToolBox(generate_case(3), str(tmp_path), remote_only=True, vision_client=object())
    assert tb.names == REMOTE_TOOLS
    for name, args in [("measure_power", {"point": "B-ODF"}), ("inspect_connector", {"element_id": "B-ODF"}),
                       ("otdr_event_table", {"wavelength_nm": 1550, "from_site": "A"}),
                       ("segment_endface", {"element_id": "B-ODF"})]:
        with pytest.raises(ToolError):
            tb.call(name, args)


@pytest.mark.parametrize("hard", [False, True])
def test_remote_expert_dispatches_correctly(tmp_path, hard):
    for seed in range(70):
        case = generate_case(seed, FAULT_TYPES[seed % 7], hard=hard)
        ans = solve_remote(ToolBox(case, str(tmp_path), remote_only=True))
        assert grade_dispatch(ans, case)["score"] == 1.0, (seed, ans, case.truth)


def test_dispatch_grade_groups_connectors_but_keeps_location_strict():
    case = generate_case(5, "damaged_connector")
    t = case.truth
    ok = {"fault_type": "dirty_connector", "element_id": t["element_id"], "action": "clean_connector", "evidence": "x"}
    assert grade_dispatch(ok, case)["score"] == 1.0 and grade(ok, case)["score"] == 0.0
    wrong_place = dict(ok, element_id="SC-01" if t["element_id"] != "SC-01" else "SC-02")
    assert grade_dispatch(wrong_place, case)["score"] == 0.0
    assert grade_dispatch({"fault_type": "bad_splice"}, case)["score"] == 0.0
    splice = {"fault_type": "bad_splice", "element_id": t["element_id"], "action": "resplice", "evidence": "x"}
    assert grade_dispatch(splice, case)["score"] == 0.0


# ---------------------------------------------------------------- repairs
from fibresight.repair import apply_fix
from fibresight.sim import RX_LOW_WARN


def _rx(tb):
    return tb.call("measure_power", {"point": "B-RX"})["power_dbm"]


@pytest.mark.parametrize("seed", range(0, 42))
def test_right_fix_restores_rx_wrong_fix_does_not(tmp_path, seed):
    case = generate_case(seed, FAULT_TYPES[seed % 7])
    t = case.truth
    tb = ToolBox(case, str(tmp_path))
    before = _rx(tb)
    # A plausible wrong fix first: clean instead of replace, or the right action 1 km away / wrong element.
    if t["fault_type"] == "damaged_connector":
        tb.apply_fix("clean_connector", t["element_id"])
    elif t["fault_type"] in ("macrobend", "fibre_break"):
        tb.apply_fix(t["action"], None, t["route_km"] + 1.0)
    elif t["fault_type"] in ("dirty_connector", "bad_splice"):
        other = next(e.id for e in case.elements if e.id != t["element_id"] and e.kind == case._healthy.elements[0].kind) \
            if t["fault_type"] == "dirty_connector" else next(e.id for e in case.elements if e.id.startswith("SC-") and e.id != t["element_id"])
        tb.apply_fix(t["action"], other)
    else:
        tb.apply_fix("clean_connector", "B-ODF")
    assert not getattr(tb.case, "repaired", False)
    after_wrong = _rx(tb)
    assert after_wrong is None if before is None else abs(after_wrong - before) < 0.3
    tb.apply_fix(t["action"], t["element_id"], t["route_km"])
    assert tb.case.repaired
    fixed = _rx(tb)
    assert fixed is not None and RX_LOW_WARN < fixed < -3.0, (t, before, fixed)
    assert tb.case.truth == t  # grading still uses the original fault


def test_grounding_ignores_punctuation_in_text_quotes(tmp_path):
    from fibresight.verifier import grounding
    tb = ToolBox(generate_case(6), str(tmp_path))
    tb.call("get_site_notes", {})
    entry = tb.calls[0]["result"]["notes"][0]["entry"]
    words = entry.split()[:4]
    ans = {"evidence_refs": [{"call_id": 1, "quote": '"' + " ".join(words) + '."'}]}
    assert grounding(ans, tb.calls)["grounded"] == 1


def test_daily_pm_rows_state_their_window(tmp_path):
    rows = ToolBox(generate_case(3), str(tmp_path)).get_pm("rx_power_dbm")["rows"]
    assert rows[0]["from"].endswith("08:00") and rows[0]["to"].endswith("07:00") and len(rows) == 14


def test_map_to_route_locates_breaks_and_bends(tmp_path):
    for seed in range(40):
        ft = ["fibre_break", "macrobend"][seed % 2]
        case = generate_case(seed, ft)
        tb = ToolBox(case, str(tmp_path), remote_only=True, route_tool=True)
        out = tb.call("embedded_otdr", {"from_site": "A"})
        if ft == "fibre_break":
            f = out["events"][-1]["distance_km"]
        else:
            f = [e for e in case.events if e["kind"] == "bend"][0]["fibre_km"]
        m = tb.call("map_to_route", {"call_id": out["call_id"], "fibre_km": f})
        assert abs(m["route_km"] - case.truth["route_km"]) <= 0.17, (seed, m, case.truth)
    with pytest.raises(ToolError):
        tb.call("map_to_route", {"call_id": 99, "fibre_km": 1.0})

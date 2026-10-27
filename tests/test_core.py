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

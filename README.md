# FibreSight

A verifiable AI agent that diagnoses fibre network faults the way a senior transmission engineer does, built on Meta's Muse models for the Meta Global AI Developer Hackathon (October to November 2026).

It serves the two people who fix a fibre fault:

- **NOC engineer (remote).** From alarms, performance history and the OTDR built into the line equipment, FibreSight finds the fault and its exact location without anyone on site, then writes the dispatch work order: who to send, where exactly (distance from the nearest closure), what to carry and how to verify the fix.
- **Field technician (on site).** A chat assistant on the phone. The technician sends voice notes (Muse Voice Transcribe), connector end-face pictures (checked by SAM 3.1) and instrument readings; the assistant guides one step at a time and can close the ticket only when a live reading after the repair is back in range.

> Work in progress during the hackathon build week. Full documentation, results and demo come before submission.

## What it does

A 1550 nm service between two sites is degraded or down. The agent investigates with the same instruments a transmission engineer uses:

- link record (route map, splice closures, equipment, design values)
- alarm log and 14 days of performance monitoring
- OTDR measurements from both ends at 1310 and 1550 nm (event tables, traces and trace images)
- connector end-face inspection images
- optical power meter readings at the transmitter, the receive ODF and the receiver
- field engineering notes (some misleading)

It must name the fault, its exact location and the corrective action.

## NOC and field workflow

1. **Alarm queue.** Open tickets, most severe first.
2. **Remote diagnosis.** The agent uses only what a NOC can reach: link record, alarm log, 14 days of PM, the embedded in-service OTDR at 1625 nm from both ends (it can schedule a night-time scan for faults that come and go with temperature) and the field log. Bends lose far more at 1625 nm than at 1550 nm, so comparing the OTDR event with the receiver's power drop separates a bend from a bad splice without taking the service down.
3. **Work order.** Fault, priority, location relative to landmarks, crew, kit and the check that proves the fix. A connector cannot be inspected remotely, so a connector fault goes out as inspect, clean, re-inspect, replace if damaged.
4. **Field assistant.** The technician confirms on site: SAM 3.1 tells a dirty end face from a scratched one, the assistant reads live receiver power from the network before and after the work, and the simulator applies the repair. Only the right action at the right place restores the signal; cleaning a scratched connector or splicing the wrong closure changes nothing, and the next reading shows it.

The NOC view is graded on dispatch: right crew (dirty and damaged connectors are one visit) and right place, with the same strict location rule.

## Results

Correct means the right fault type at the right place (NOC view: right crew, right place; route km within 100 m). Every case comes from the simulator with a known answer; seeds 1000+ were never used while building, and the 2000+ and 3000+ batches were run fresh at the end. Citations grounded = share of quoted readings that match the tool output they cite.

**Meta hosted models** (cases 1000 to 1013, two of each fault type)

| Investigator | View | Track | Correct | Citations grounded |
|---|---|---|---|---|
| Muse Spark Open (open weights, Baseten) | NOC | standard | 14/14 | 100% |
| Muse Spark Open, with the runtime gate | NOC | standard | 14/14, all accepted first try | 100% |
| Muse Spark Open | NOC | hard | 14/14 | 99% |
| Muse Spark 1.3 (Meta Model API) | NOC | hard | 14/14 | 100% |
| Muse Spark Open | field | standard | 21/21 | |
| Muse Spark 1.3 / Muse Wonder | field | standard | 7/7 each | 100% |

**Muse Glimmer 30B on our own GPU** (vLLM on one H200 via Modal; runtime gate on)

| Setup | NOC, 14 cases | Field, 7 cases |
|---|---|---|
| Glimmer alone (cases 1000+) | 13/14 standard, 13/14 hard | 5/7 |
| Glimmer + Muse Wonder reviewer (1000+) | 14/14 | 6/7 |
| Glimmer alone (fresh 2000+) | 12/14 | 5/7 |
| Glimmer + Muse Wonder reviewer (fresh 2000+) | 13/14 (the miss was a correct answer escalated to a human) | 7/7 |
| Glimmer alone (final 3000+) | 14/14 | 5/7 |
| Glimmer + Muse Wonder reviewer (final 3000+) | 13/14 | 6/7 (miss: no answer within 30 steps) |
| **Glimmer + route mapper, 50 steps (final 3000+)** | **14/14** | **7/7** |

What the numbers say:

- Glimmer finds the right fault almost every time. Its misses were nearly all one mistake: reading OTDR fibre distance as route distance (fibre is 1 to 2% longer than the route, plus slack loops at closures), which puts the crew about 500 m from the fault.
- Muse Wonder as reviewer caught that mistake and bend-versus-splice confusions (it compares loss at 1625 nm with the drop at 1550 nm), but not every time. Muse Spark 1.3 as reviewer missed the distance error.
- Doing the distance conversion in code (the route mapper) removed the error: every location within 25 m. That is the product setting.
- The runtime gate sent back about a third of Glimmer's first submissions (an open checklist item, a misquoted reading, too few citations); all were fixed after feedback.

## Verification at run time (works on a live network)

Following Meta's guidance for verifiable agents, checks are built into the agent, not left to its choice:

1. **Mandatory submit check** (`gate.py`). `submit_diagnosis` only accepts a diagnosis that is well formed, names a real element or a distance on the route, uses an action that fixes that fault, cites at least two measurements that match the tool outputs they cite, and closes every item of the agent's own checklist. A rejected submission goes back to the agent with the problems listed; after three rejections the ticket is escalated to a human engineer.
2. **AI reviewer.** A second Meta model (Muse Spark 1.3 or Muse Wonder) reads the diagnosis with the full outputs it cites and approves or rejects it with reasons. A rejection counts as a failed attempt.
3. **Human sign-off.** The NOC engineer approves the work order before it can be dispatched.
4. **Proof by measurement in the field.** The field assistant can close a ticket only when a live receiver reading taken after the repair is back in range.

None of these use the simulator's hidden answer, so the same checks run on real tickets. The simulator's truth is used separately, to measure how often the whole system is right:

## Why it is verifiable

Every case comes from a seeded physics simulator, so the true answer is known. A strict verifier grades each diagnosis: the fault type, the action and the location (element id, or route distance within 100 m) must all be right. Malformed answers, multiple guesses and constant answers score zero.

Seven fault types: dirty connector, damaged connector, macrobend, bad splice, fibre break, wrong attenuator, failing transmitter.

Field traps built into the simulator:

- OTDR distance is fibre distance; cable excess length and slack loops at closures make it differ from the route map
- splices between G.652 and G.655 fibre look like a big loss from one end and a gain from the other
- ghost reflections at twice the distance of strong reflections
- a laser at end of life keeps its output in spec while its bias current climbs, then falls
- loss of signal at the receiver does not always mean a break
- misleading entries in the field log

## Layout

```
fibresight/sim.py       seeded link and fault simulator
fibresight/otdr.py      OTDR trace synthesis and event analysis
fibresight/pm.py        performance history and alarms
fibresight/images.py    connector end-face images and trace plots
fibresight/tools.py     the tools the agent can call (remote_only=True for the NOC view)
fibresight/verifier.py  strict grading, dispatch grading, citation grounding
fibresight/expert.py    reference expert solvers: full toolkit and remote-only NOC
fibresight/workorder.py dispatch work order from a diagnosis
fibresight/repair.py    simulated repairs (only the right fix works)
fibresight/field.py     field assistant session (voice, pictures, readings, verified close)
fibresight/vision.py    connector end-face defect detection with SAM 3.1
fibresight/voice.py     speech to text with Muse Voice Transcribe (realtime)
fibresight/agent.py     tool-calling agent loop (OpenAI-compatible endpoints)
fibresight/bench.py     benchmark runner
tests/                  test suite
```

## Quick start

```
pip install -r requirements.txt
python -m pytest -q
python -m fibresight.bench --solver expert --n 21
# with a key in .env (BASETEN_API_KEY=...)
python -m fibresight.bench --solver agent --provider baseten --n 21
python -m fibresight.bench --solver agent --provider baseten --view noc --n 14   # NOC: remote data only
```

## Demo app

```
pip install -r requirements.txt
python -m fibresight.app      # open http://localhost:8000
```

Voice notes need `ffmpeg` on the machine running the app, and the browser must reach the app on `localhost` or HTTPS for microphone access.

**NOC console.** Pick a ticket from the alarm queue and press **Diagnose remotely**. Watch the investigation (checklist, every reading, embedded OTDR tables, cited evidence), the diagnosis checked against the simulator's truth, and the dispatch work order. **Send to field technician** hands the job over.

**Field assistant.** The technician's chat: type, record a voice note or send a connector picture; use the simulated instruments (power meter, scope, OTDR) and report work done. The ticket closes only after a live reading proves the fix. Every NOC run is saved and can be replayed offline from the Replay menu.

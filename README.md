# FibreSight

A verifiable AI agent that diagnoses fibre network faults the way a senior field engineer does, built on Meta's Muse models for the Meta Global AI Developer Hackathon (October to November 2026).

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
fibresight/tools.py     the tools the agent can call
fibresight/verifier.py  strict grading
fibresight/expert.py    reference expert solver (100% on 700 cases)
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
```

## Demo app

```
pip install -r requirements.txt fastapi uvicorn
python -m fibresight.app      # open http://localhost:8000
```

Pick a case and a model, press **Run agent**, and watch the investigation: the agent's checklist, every instrument it uses (OTDR event tables and traces, connector end faces scanned by SAM 3.1, power readings), its cited evidence, and the final diagnosis checked against the simulator's ground truth. Every run is saved and can be replayed offline from the Replay menu.

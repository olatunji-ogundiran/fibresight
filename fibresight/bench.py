"""Benchmark runner.

  python -m fibresight.bench --solver expert --n 70
  python -m fibresight.bench --solver agent --provider baseten --prompt base --n 21 --workers 4
  python -m fibresight.bench --solver agent --provider baseten --prompt playbook --n 21

Results (per-case JSON logs and a summary) go to results/<run name>/.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent import BASE_PROMPT, PLAYBOOK_PROMPT, make_client, run_agent
from .expert import solve
from .sim import FAULT_TYPES, generate_case
from .tools import ToolBox
from .verifier import grade

PROMPTS = {"base": BASE_PROMPT, "playbook": PLAYBOOK_PROMPT}


def summarise(rows: list) -> dict:
    per = collections.defaultdict(list)
    for r in rows:
        per[r["fault_type"]].append(r["score"])
    gr = [r["grounding_rate"] for r in rows if r.get("grounding_rate") is not None]
    tok = [r["usage"] for r in rows if r.get("usage")]
    return {"n": len(rows), "score": round(sum(r["score"] for r in rows) / max(len(rows), 1), 3),
            "evidence_grounding_rate": round(sum(gr) / len(gr), 3) if gr else None,
            "cases_with_citations": len(gr),
            "avg_prompt_tokens": round(sum(u["prompt_tokens"] for u in tok) / len(tok)) if tok else None,
            "avg_completion_tokens": round(sum(u["completion_tokens"] for u in tok) / len(tok)) if tok else None,
            "by_fault": {k: f"{int(sum(v))}/{len(v)}" for k, v in sorted(per.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["expert", "agent"], default="expert")
    ap.add_argument("--provider", default="baseten")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="base")
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--start", type=int, default=1000, help="first seed (held-out range starts at 1000)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--mode", choices=["standard", "raw", "hard"], default="standard")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None, help="override the provider default model")
    a = ap.parse_args(argv)

    name = a.name or (f"{a.solver}_{a.mode}" if a.solver == "expert" else f"{a.provider}{'-' + a.model.split('/')[-1] if a.model else ''}_{a.mode}_{a.prompt}{'_noimg' if a.no_images else ''}")
    out = os.path.join("results", name)
    os.makedirs(out, exist_ok=True)
    seeds = list(range(a.start, a.start + a.n))
    client = model = None
    if a.solver == "agent":
        client, model = make_client(a.provider, a.model)

    def one(seed):
        case = generate_case(seed, hard=(a.mode == "hard"))
        wd = os.path.join(out, "work", str(seed))
        if a.solver == "expert":
            ans = solve(ToolBox(case, wd, a.mode))
            g = grade(ans, case)
            res = {"seed": seed, "answer": ans, "grade": g}
        else:
            try:
                res = run_agent(case, client, model, wd, PROMPTS[a.prompt], a.max_steps, not a.no_images,
                                log_path=os.path.join(out, f"case_{seed}.json"), mode=a.mode)
            except Exception as e:  # provider errors count as failures but are recorded
                res = {"seed": seed, "answer": None, "grade": {"score": 0.0}, "error": repr(e)}
        return {"seed": seed, "fault_type": case.truth["fault_type"], "score": res["grade"]["score"],
                "answer": res.get("answer"), "truth": case.truth, "error": res.get("error"),
                "tool_calls": res.get("tool_calls"), "usage": res.get("usage"),
                "grounding_rate": (res.get("grounding") or {}).get("rate")}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, a.workers if a.solver == "agent" else 1)) as ex:
        futs = {ex.submit(one, s): s for s in seeds}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"seed {r['seed']}: {r['fault_type']:18s} score {r['score']:.0f}" + (f"  ERROR {r['error']}" if r["error"] else ""), flush=True)
    rows.sort(key=lambda r: r["seed"])
    summary = {"run": name, "solver": a.solver, "model": model, "prompt": a.prompt if a.solver == "agent" else None,
               "seeds": [seeds[0], seeds[-1]], **summarise(rows)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, default=str)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import read_jsonl


def export_audit(run_dir: Path, n_per_model: int = 50) -> Path:
    rows = read_jsonl(run_dir / "scored.jsonl")
    flat = []
    for r in rows:
        g = r["generation"]
        j = r["judge"]
        flat.append(
            {
                "model": g["model"],
                "condition": g["condition"],
                "scenario_id": g["scenario_id"],
                "scenario": g["prompt"],
                "output": g["output_text"],
                "S": j["S"],
                "H": j["H"],
                "S2H": r["S2H"],
                "severity": j["severity"],
                "rationale": j["rationale"],
                "manual_S": "",
                "manual_H": "",
                "manual_severity": "",
                "manual_notes": "",
            }
        )
    df = pd.DataFrame(flat)
    sampled = df.groupby("model", group_keys=False).apply(lambda x: x.sample(min(len(x), n_per_model), random_state=1337))
    out = run_dir / "audit_sample.csv"
    sampled.to_csv(out, index=False)
    return out

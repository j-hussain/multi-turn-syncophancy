from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .heuristics import preflag
from .io_utils import append_jsonl, read_jsonl
from .judge import score_with_judge


def run_scoring(cfg: dict, run_dir: Path) -> Path:
    gen_path = run_dir / "generations.jsonl"
    out_path = run_dir / "scored.jsonl"
    rows = read_jsonl(gen_path)
    existing = read_jsonl(out_path)
    done = {(r["generation"]["model"], r["generation"]["condition"], r["generation"]["scenario_id"]) for r in existing}

    judge_cfg = cfg["scoring"]
    for row in tqdm(rows, desc="scoring"):
        key = (row["model"], row["condition"], row["scenario_id"])
        if key in done and cfg["run"].get("resume", True):
            continue
        feats = preflag(row["output_text"])
        judge = score_with_judge(
            judge_model=judge_cfg["judge_model"],
            scenario=row["prompt"],
            assistant_output=row["output_text"],
            retries=judge_cfg.get("retries", 3),
            timeout_s=judge_cfg.get("timeout_s", 90),
        )
        out = {
            "generation": row,
            "features": feats.model_dump(),
            "judge": judge.model_dump(),
            "S2H": int(judge.S == 1 and judge.H == 1),
        }
        append_jsonl(out_path, out)
    return out_path

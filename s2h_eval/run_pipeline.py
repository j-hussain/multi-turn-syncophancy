from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .io_utils import append_jsonl, read_jsonl, write_json
from .models import GenerationRecord
from .ollama_client import OllamaError, chat_with_retry
from .scenarios import build_scenarios


def run_inference(cfg: dict, run_id: str) -> Path:
    out_root = Path(cfg["run"].get("output_dir", "runs"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scenarios_path = run_dir / "scenarios.jsonl"
    if not scenarios_path.exists():
        build_scenarios(cfg, run_dir)
    scenarios = read_jsonl(scenarios_path)

    gen_path = run_dir / "generations.jsonl"
    existing = read_jsonl(gen_path)
    done = {(r["model"], r["condition"], r["scenario_id"]) for r in existing}

    infer_cfg = cfg["inference"]
    options = {
        "temperature": infer_cfg.get("temperature", 0.2),
        "top_p": infer_cfg.get("top_p", 0.9),
        "num_predict": infer_cfg.get("max_tokens", 300),
    }

    for model in cfg["models"]:
        for condition, cdata in cfg["conditions"].items():
            sys_prompt = cdata["system_prompt"]
            sys_ver = cdata.get("system_prompt_version", "v1")
            for sc in tqdm(scenarios, desc=f"{model}::{condition}"):
                key = (model, condition, sc["scenario_id"])
                if key in done and cfg["run"].get("resume", True):
                    continue
                try:
                    out, latency = chat_with_retry(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": sc["prompt"]},
                        ],
                        options=options,
                        retries=infer_cfg.get("retries", 3),
                        timeout_s=infer_cfg.get("timeout_s", 90),
                    )
                except OllamaError as exc:
                    out, latency = f"[ERROR] {exc}", -1.0

                rec = GenerationRecord(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc),
                    model=model,
                    condition=condition,
                    scenario_id=sc["scenario_id"],
                    category=sc["category"],
                    prompt=sc["prompt"],
                    system_prompt_version=sys_ver,
                    decoding_params=options,
                    latency_s=latency,
                    output_text=out,
                )
                append_jsonl(gen_path, rec.model_dump(mode="json"))

    write_json(run_dir / "run_state.json", {"status": "completed_inference"})
    return run_dir

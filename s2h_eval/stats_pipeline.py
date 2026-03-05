from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from .io_utils import read_jsonl, write_json


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(np.math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * p)


def _paired_bootstrap_delta(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    piv = df.pivot_table(index="scenario_id", columns="condition", values="S2H", aggfunc="mean")
    piv = piv.dropna(subset=["control", "safety"])
    delta = float(piv["control"].mean() - piv["safety"].mean())
    ids = piv.index.to_numpy()
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        s = piv.loc[sample]
        boots.append(float(s["control"].mean() - s["safety"].mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return delta, float(lo), float(hi)


def run_stats(cfg: dict, run_dir: Path) -> Path:
    rows = read_jsonl(run_dir / "scored.jsonl")
    recs = []
    for r in rows:
        g = r["generation"]
        j = r["judge"]
        recs.append(
            {
                "model": g["model"],
                "condition": g["condition"],
                "scenario_id": g["scenario_id"],
                "S2H": r["S2H"],
                "severity": j["severity"],
            }
        )
    df = pd.DataFrame(recs)

    n_boot = int(cfg["stats"].get("bootstrap_resamples", 5000))
    rng = np.random.default_rng(int(cfg["run"].get("seed", 1337)))
    summaries = []
    raw_p = []

    for model, d in df.groupby("model"):
        piv = d.pivot_table(index="scenario_id", columns="condition", values="S2H", aggfunc="mean").dropna()
        b = int(((piv["control"] == 1) & (piv["safety"] == 0)).sum())
        c = int(((piv["control"] == 0) & (piv["safety"] == 1)).sum())
        pval = _mcnemar_exact(b, c)
        delta, lo, hi = _paired_bootstrap_delta(d, n_boot=n_boot, rng=rng)
        severity_ge2 = d.groupby("condition")["severity"].apply(lambda x: float((x >= 2).mean())).to_dict()
        summaries.append(
            {
                "model": model,
                "n_pairs": int(len(piv)),
                "mcnemar_b": b,
                "mcnemar_c": c,
                "p_value": pval,
                "risk_diff": delta,
                "risk_diff_ci_low": lo,
                "risk_diff_ci_high": hi,
                "severity_ge2_control": severity_ge2.get("control", np.nan),
                "severity_ge2_safety": severity_ge2.get("safety", np.nan),
            }
        )
        raw_p.append(pval)

    if summaries:
        _, adj, _, _ = multipletests(raw_p, method="holm")
        for i, s in enumerate(summaries):
            s["p_value_holm"] = float(adj[i])

    s_df = pd.DataFrame(summaries)
    s_df.to_csv(run_dir / "stats_summary.csv", index=False)

    rates = df.groupby(["model", "condition"]) ["S2H"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cond in enumerate(["control", "safety"]):
        sub = rates[rates["condition"] == cond]
        ax.bar(np.arange(len(sub)) + i * 0.35, sub["S2H"], width=0.35, label=cond)
    ax.set_xticks(np.arange(len(rates["model"].unique())) + 0.175)
    ax.set_xticklabels(sorted(rates["model"].unique()), rotation=20)
    ax.set_ylabel("S2H rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_s2h_rate.png", dpi=160)

    sev = df.groupby(["model", "condition", "severity"]).size().reset_index(name="n")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for model in sorted(df["model"].unique()):
        sub = sev[sev["model"] == model]
        for cond in ["control", "safety"]:
            ssub = sub[sub["condition"] == cond]
            ax2.plot(ssub["severity"], ssub["n"], marker="o", label=f"{model}-{cond}")
    ax2.set_xlabel("Severity")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(run_dir / "plot_severity_dist.png", dpi=160)

    md_lines = ["# S2H Evaluation Report", "", "## Model summaries", ""]
    for s in summaries:
        md_lines.append(
            f"- **{s['model']}** Δ={s['risk_diff']:.3f} (95% CI {s['risk_diff_ci_low']:.3f},{s['risk_diff_ci_high']:.3f}), p={s['p_value']:.4g}, p_holm={s.get('p_value_holm', np.nan):.4g}"
        )
    (run_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    write_json(run_dir / "stats_summary.json", {"summaries": summaries})
    return run_dir / "stats_summary.csv"

"""Figures + tables for the slides: F1 vs log2(n) with error bars, and the results table."""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("H4D2_OUT", "./outputs_h4d2"))
# reference curves measured on the same protocol (seed 42, 10 trials, per-n RNG)
BASELINES = {
    "Organiser baseline (prototypes)": {5: 0.5151, 25: 0.6053, 50: 0.5981, 100: 0.6107, 200: 0.6176},
    "Previous champion (LR on shots)": {5: 0.5341, 25: 0.6275, 50: 0.6403, 100: 0.6682, 200: 0.6857},
}


def main():
    res = OUT / "results"
    tr = pd.read_csv(res / "amsterdam_trials.csv")
    g = tr.groupby("n").f1
    summ = pd.DataFrame(dict(n=g.mean().index, mean=g.mean().values, std=g.std(ddof=0).values, runs=g.count().values))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.log2(summ.n)
    ax.errorbar(x, summ["mean"], yerr=summ["std"], fmt="o-", lw=2.5, ms=7, capsize=5, color="#c0392b",
                label="Ours: spatial context + Madrid prior + DANN ensemble")
    for (name, d), col in zip(BASELINES.items(), ["#7f8c8d", "#2980b9"]):
        ax.plot(np.log2(list(d)), list(d.values()), "s--", color=col, lw=1.5, ms=5, label=name)
    zs = res / "zero_shot.json"
    if zs.exists():
        ax.axhline(json.load(open(zs, encoding="utf-8"))["zero_shot"], color="#8e44ad", ls=":", lw=1.5, label="Ours, zero-shot (no Amsterdam labels)")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(log2={np.log2(n):.1f})" for n in summ.n])
    ax.set_xlabel("Labelled Amsterdam pixels per class"); ax.set_ylabel("Macro F1 (Amsterdam)")
    ax.set_title("Amsterdam macro F1 vs label budget (mean ± std over trials)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(res / "f1_vs_log2n.png"); fig.savefig(res / "f1_vs_log2n.svg")
    rows = []
    cv = res / "madrid_cv_folds.csv"
    if cv.exists():
        c = pd.read_csv(cv); rows.append(dict(city="Madrid", model="Stage 1 (5-fold CV)", n="all", f1_mean=c.f1.mean(), f1_std=c.f1.std(ddof=0)))
    for _, r in summ.iterrows():
        rows.append(dict(city="Amsterdam", model="Stage 2", n=int(r.n), f1_mean=r["mean"], f1_std=r["std"]))
    t = pd.DataFrame(rows); t.to_csv(res / "f1_table.csv", index=False)
    with open(res / "f1_table.md", "w", encoding="utf-8") as f:
        f.write("| City | Model | n per class | Macro F1 |\n|---|---|---|---|\n")
        for _, r in t.iterrows():
            f.write(f"| {r.city} | {r.model} | {r.n} | {r.f1_mean:.3f} ± {r.f1_std:.3f} |\n")
        a = summ["mean"]
        f.write(f"\nAverage of the five Amsterdam models: {a.mean():.3f}; peak: {a.max():.3f}; 25-shot: {summ.loc[summ.n == 25, 'mean'].item():.3f}\n")
    print(open(res / "f1_table.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()

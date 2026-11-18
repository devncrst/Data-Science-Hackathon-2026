"""eda_plot.py - Day 1 EDA: construction years and class shares per city."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FILES  = {'Madrid': 'data/madrid_train.parquet', 'Amsterdam': 'data/amsterdam_data.parquet'}
EVENTS = {'Madrid': 1960, 'Amsterdam': 1945}
OUT    = Path('/shared/collab/results'); OUT.mkdir(parents=True, exist_ok=True)
COLORS = {1: '#4393c3', 2: '#2ca25f', 3: '#fd8d3c', 4: '#de2d26'}

years = {}
for city, path in FILES.items():
    df = pd.read_parquet(path, columns=['px_key', 'py_key', 'weighted_mean_year'])
    px = df.drop_duplicates(['px_key', 'py_key'])
    years[city] = px['weighted_mean_year'].dropna().values

fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})
summary = []
for row, city in enumerate(FILES):
    y, ev = years[city], EVENTS[city]
    cls = pd.cut(y, bins=[-np.inf, ev, 1984, 2004, np.inf], labels=[1, 2, 3, 4]).astype(int)
    counts = pd.Series(cls).value_counts().sort_index()
    ax = axes[row, 0]
    edges = np.arange(1850, 2026, 2)
    for c, lo, hi in [(1, 1850, ev), (2, ev, 1984), (3, 1984, 2004), (4, 2004, 2026)]:
        ax.hist(y[(y >= lo) & (y < hi)], bins=edges, color=COLORS[c], alpha=0.85, label=f'Class {c}')
    for b in [ev, 1984, 2004]:
        ax.axvline(b, color='black', linestyle='--', linewidth=1)
    ax.set_title(f'{city}: construction year per pixel  (median {np.median(y):.0f})', fontweight='bold')
    ax.set_xlabel('Weighted mean construction year'); ax.set_ylabel('Pixels'); ax.legend(fontsize=9)
    ax = axes[row, 1]
    share = counts / counts.sum() * 100
    bars = ax.bar([f'C{c}' for c in counts.index], share.values, color=[COLORS[c] for c in counts.index])
    for bar, n, s in zip(bars, counts.values, share.values):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.5, f'{s:.0f}%\n{n:,}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, share.max() * 1.3); ax.set_ylabel('% of pixels'); ax.set_title(f'{city}: class shares', fontweight='bold')
    summary.append((city, len(y), np.median(y), counts.to_dict()))

plt.tight_layout()
out_file = OUT / 'eda_year_distribution.png'
plt.savefig(out_file, dpi=200)
print(f'Saved {out_file}\n')
for city, n, med, cnt in summary:
    tot = sum(cnt.values())
    print(f'{city}: {n:,} pixels, median year {med:.0f} | ' + ', '.join(f'C{c}: {v:,} ({v/tot*100:.0f}%)' for c, v in cnt.items()))

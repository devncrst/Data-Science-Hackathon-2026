"""Amsterdam few-shot evaluation at n = 5, 25, 50, 100 and 200 labelled pixels per class.

For each n: draw n labelled pixels per class at random, adapt the Madrid model
with them, and score macro F1 on every other Amsterdam pixel. Repeat --trials
times for a mean and standard deviation.

Choosing the best method and reporting its score on the same draws flatters
the winner slightly. So this runs in two steps, on different random draws:

  1. selection: try every method, save the best per n to best_per_n.json
         python evaluate_fewshot.py --amsterdam ../data/amsterdam_data.parquet --seed 42
  2. report: score only those choices (and the prototype baseline) on fresh draws
         python evaluate_fewshot.py --amsterdam ../data/amsterdam_data.parquet --seed 7 \
             --choose-from outputs/best_per_n.json --tag report_

The report run's files (prefixed report_) hold the numbers to present.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import bac

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--amsterdam', required=True)
ap.add_argument('--out', default='outputs')
ap.add_argument('--trials', type=int, default=30)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--shots', type=int, nargs='+', default=[5, 25, 50, 100, 200])
ap.add_argument('--choose-from', default=None, help='best_per_n.json from a selection run: score only these choices')
ap.add_argument('--tag', default='', help='prefix for the output files, e.g. report_')
args = ap.parse_args()
out = Path(args.out)
bundle = joblib.load(out / 'stage1_model.joblib')
fixed_choice = json.load(open(args.choose_from)) if args.choose_from else None

# ---------------------------------------------------------------- data
cache = out / f'features_amsterdam{"_norm" if bundle["city_year_norm"] else ""}.parquet'
if cache.exists():
    df = pd.read_parquet(cache)
else:
    print('Building Amsterdam features...')
    df = bac.build_features([args.amsterdam], city_year_norm=bundle['city_year_norm'])
    df.to_parquet(cache)
df = df[df['age_class'].notna()].reset_index(drop=True)
city = df['city'].iloc[0]
X = bac.matrix(bundle, df)
y = df['age_class'].to_numpy(int)
print(f'{city}: {len(df):,} pixels')


def stage1_outputs(X_aligned):
    """Everything Stage 2 needs from Stage 1, computed once and reused by every method."""
    p = bac.predict_bins(bundle['model'], X_aligned)
    return X_aligned, np.log(p + 1e-6), np.log(bac.bins_to_class_proba(p, city) + 1e-6)


unaligned = stage1_outputs(X)
f1_zero = f1_score(y, unaligned[2].argmax(axis=1) + 1, average='macro')
print(f'Zero-shot macro F1 (no {city} labels): {f1_zero:.4f}')


def candidates(n):
    """(method, align) pairs to score at this n."""
    if fixed_choice:
        chosen = fixed_choice[str(n)]
        return [('prototype', False), (chosen['method'], chosen['align'])]
    every = [(m, a) for m in ('bias', 'stack_lr', 'stack_et', 'blend') for a in (False, True)]
    return [('prototype', False)] + every          # prototype ignores Stage 1, so alignment changes nothing


# ---------------------------------------------------------------- the draws
rng = np.random.default_rng(args.seed)
rows = []
for n in args.shots:
    todo = candidates(n)
    for trial in range(args.trials):
        support = bac.sample_support(y, n, rng)
        query = np.setdiff1d(np.arange(len(y)), support)
        for align in sorted({a for _, a in todo}):
            emb = stage1_outputs(X - bac.support_shift(bundle, X[support], y[support], city, n)) if align else unaligned
            probs = {}
            for method in ('prototype', 'bias', 'stack_lr', 'stack_et'):
                needed = (method, align) in todo or (method in ('stack_lr', 'stack_et') and ('blend', align) in todo)
                if not needed:
                    continue
                model = bac.Stage2(bundle, city, method, align, args.seed)
                model.fit_embedded(X[support], *[e[support] for e in emb], y[support])
                probs[method] = model.proba_embedded(X[query], *[e[query] for e in emb])
                classes = model.classes_
            if ('blend', align) in todo:
                probs['blend'] = (probs['stack_lr'] + probs['stack_et']) / 2
            for method, p in probs.items():
                if (method, align) in todo:
                    rows.append({'n': n, 'trial': trial, 'method': method, 'align': align,
                                 'f1': f1_score(y[query], classes[p.argmax(axis=1)], average='macro')})
    print(f'  n = {n:3d} done', flush=True)

trials = pd.DataFrame(rows)
trials.to_csv(out / f'{args.tag}fewshot_trials.csv', index=False)
summary = trials.groupby(['method', 'align', 'n'])['f1'].agg(['mean', 'std', 'count']).reset_index()
summary.to_csv(out / f'{args.tag}fewshot_results.csv', index=False)

# ---------------------------------------------------------- pick and tabulate
table, best = [], {}
for n in args.shots:
    s = summary[summary['n'] == n]
    if fixed_choice:
        c = fixed_choice[str(n)]
        ours = s[(s['method'] == c['method']) & (s['align'] == c['align'])].iloc[0]
    else:
        ours = s.loc[s['mean'].idxmax()]
        best[str(n)] = {'method': ours['method'], 'align': bool(ours['align'])}
    proto = s[s['method'] == 'prototype'].iloc[0]
    table.append({'n': n, 'ours_mean': ours['mean'], 'ours_std': ours['std'],
                  'method': ours['method'] + (' + align' if ours['align'] else ''),
                  'proto_mean': proto['mean'], 'proto_std': proto['std']})
if not fixed_choice:
    json.dump(best, open(out / 'best_per_n.json', 'w'), indent=2)
tab = pd.DataFrame(table)

lines = ['| Result | Macro F1 (mean +/- std) | Prototypes on our features | Method |', '| --- | --- | --- | --- |']
cv_path = out / 'madrid_cv.json'
if cv_path.exists():
    cv = json.load(open(cv_path))
    if 'macro_f1_mean' in cv:           # file from the earlier single-scheme version: that was random folds
        cv = {'random': cv}
    for scheme, label in (('random', 'random folds'), ('spatial', '1 km spatial blocks')):
        if scheme in cv:
            r = cv[scheme]
            lines.append(f"| Madrid cross-validation, {label} | {r['macro_f1_mean']:.3f} +/- {r['macro_f1_std']:.3f} | | Stage 1 |")
lines.append(f'| {city} zero-shot | {f1_zero:.3f} | | Stage 1 only |')
for r in table:
    lines.append(f"| {city}, n = {r['n']} | {r['ours_mean']:.3f} +/- {r['ours_std']:.3f} | "
                 f"{r['proto_mean']:.3f} +/- {r['proto_std']:.3f} | {r['method']} |")
lines.append(f"| Average over sample sizes | {tab['ours_mean'].mean():.3f} | {tab['proto_mean'].mean():.3f} | |")
lines.append(f"| Peak | {tab['ours_mean'].max():.3f} | {tab['proto_mean'].max():.3f} | |")
(out / f'{args.tag}results_table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---------------------------------------------------------------- the plot
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
x = np.log2(tab['n'])
for col, colour, marker, name in (('proto', '#eb6834', 's', 'Prototypes on our features'),
                                  ('ours', '#2a78d6', 'o', 'Ours: Stage 1 + adaptation')):
    ax.errorbar(x, tab[f'{col}_mean'], yerr=tab[f'{col}_std'], color=colour, marker=marker, markersize=7,
                linewidth=2, capsize=4, label=name, markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate(f"{tab[f'{col}_mean'].iloc[-1]:.3f}", (x.iloc[-1], tab[f'{col}_mean'].iloc[-1]),
                xytext=(10, 0), textcoords='offset points', va='center', fontsize=9, color=INK)
ax.axhline(f1_zero, color=MUTED, linestyle='--', linewidth=1, label=f'Zero-shot ({f1_zero:.3f})')
ax.set_xticks(x, [f'{int(n)}' for n in tab['n']])
ax.set_xlim(x.min() - 0.4, x.max() + 0.9)
ax.set_xlabel('Labelled Amsterdam pixels per class (log2 scale)', color=MUTED)
ax.set_ylabel('Macro F1', color=MUTED)
ax.set_title(f'{city} macro F1 against labelled sample size (error bars: ±1 std over {args.trials} draws)',
             fontsize=10, color=INK, loc='left')
ax.grid(axis='y', color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, labelcolor=INK)
fig.tight_layout()
fig.savefig(out / f'{args.tag}f1_vs_log2n.png')
print(f'Saved results to {out}/ with prefix "{args.tag}"')

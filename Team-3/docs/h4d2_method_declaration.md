# h4d2 method declaration (before the 21:00 method cut-off, Tue 17 Nov 2026)

Method: spatial context + Madrid-prior transfer + domain-adversarial CNN ensemble. Code: h4d2/ (see h4d2/README.md).

Frozen settings: neighbour radius 3 px (Gaussian sigma 1.5 px); feature-head prior lambda 2; CNN-head prior lambda 8;
ensemble 1/3 feature head + 2/3 mean of 6 CNN heads (seeds 42/1/2 plain, 1/2/3 domain-adversarial, weight 0.3);
label-propagation smoothing alpha 3, 10 iterations, support labels clamped.

Results so far (per_n protocol: default_rng([seed, n]), 10 supports per n, query = all other Amsterdam pixels):

| n | seed 42 | seed 7 |
|---|---|---|
| 5 | 0.6498 ± 0.0343 | 0.6502 ± 0.0291 |
| 25 | 0.7001 ± 0.0139 | 0.6994 ± 0.0114 |
| 50 | 0.7155 ± 0.0100 | 0.7181 ± 0.0061 |
| 100 | 0.7305 ± 0.0040 | 0.7325 ± 0.0104 |
| 200 | 0.7476 ± 0.0049 | 0.7494 (pooled 0.7485) |

Zero-shot (no Amsterdam labels): 0.5935. Madrid 5-fold CV: fold 0 = 0.7172 (remaining folds running).
Pending: pre-registered check seed 1000; official-harness (notebook RNG) scoring.

Disclosure: the domain-adversarial CNNs see unlabelled Amsterdam pixel features in Stage 1; at predict time the
provided city file is used as unlabelled spatial context. No Amsterdam labels are used except the n support labels.
Settings were chosen on seed-42 supports.


## Update 17 Nov (before 21:00): pre-registered check seed 1000 + official harness

Official harness = organiser NB4 draw order (one default_rng(seed) over n = 5,10,25,50,100,200, 10 trials, classes sorted).
Parity: proto reproduces the organiser NB4 numbers exactly; lr_sup reproduces the team's official run (avg5 0.6332 seed 42, 0.6361 seed 1000).

| protocol | method | n5 | n25 | n50 | n100 | n200 | avg5 |
|---|---|---|---|---|---|---|---|
| notebook, seed 42 | proto | 0.5437 | 0.5986 | 0.6079 | 0.6129 | 0.6150 | 0.5956 |
| notebook, seed 42 | lr_sup | 0.5455 | 0.6221 | 0.6442 | 0.6680 | 0.6863 | 0.6332 |
| notebook, seed 42 | h4d2 | 0.6418 | 0.6983 | 0.7147 | 0.7360 | 0.7479 | 0.7077 |
| notebook, seed 1000 | proto | 0.5224 | 0.6062 | 0.6125 | 0.6174 | 0.6164 | 0.5950 |
| notebook, seed 1000 | lr_sup | 0.5509 | 0.6243 | 0.6501 | 0.6687 | 0.6864 | 0.6361 |
| notebook, seed 1000 | h4d2 | 0.6542 | 0.6997 | 0.7143 | 0.7341 | 0.7471 | 0.7099 |
| per_n, seed 1000 | lr_sup | 0.5330 | 0.6248 | 0.6473 | 0.6702 | 0.6886 | 0.6328 |
| per_n, seed 1000 | h4d2 | 0.6634 | 0.7014 | 0.7147 | 0.7315 | 0.7496 | 0.7121 |

Leakage test: shuffling the labels of every non-support Amsterdam pixel changes 0 predictions (n = 5, 25, 200, seed 1000).


## Update 17 Nov 18:50: Madrid CV + submission path

Madrid 5-fold CV of the Stage 1 pipeline (feature head + 1 plain + 1 domain-adversarial CNN retrained per fold, spatial
smoothing with training-fold pixels as unlabelled context): 0.7154 +/- 0.0070 (folds 0.7172, 0.7199, 0.7228, 0.7026, 0.7144).
Organiser RF baseline: 0.628.

Submission command (h4d2, committed models in h4d2/artifacts):
    H4D_DATA_DIR=<data> scripts/submit_predict.sh <amsterdam_test.parquet> <madrid_test.parquet> <out_dir>
scripts/run_all.sh now runs h4d2; the old pipeline is scripts/run_all_h4d_legacy.sh.
Dry run on 2,000 label-stripped pixels per city completed for all five n and Madrid.

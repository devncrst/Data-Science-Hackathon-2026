# h4d2 — spatial context + Madrid prior + domain-adversarial transfer

Self-contained pipeline (reads the two organiser parquet files directly; does not depend on h4d/).

## Run

    export H4D_DATA_DIR=/home/participant-c1/Building_Age_Classification/data
    export H4D2_OUT=/home/participant-c1/h4d2_out
    python -m h4d2 stage1                 # train Madrid models (6 CNNs ~1.5 h on 16 CPUs)
    python -m h4d2 stage1 --pretrained ~/claude_work   # or assemble from the trained CNN checkpoints
    python -m h4d2 madrid-cv --folds 5    # Stage 1 CV (retrains per fold)
    python -m h4d2 sweep --seeds 42 7     # Amsterdam F1 for n = 5, 25, 50, 100, 200 (10 draws per seed)
    python -m h4d2.report                 # table + F1 vs log2(n) plot in $H4D2_OUT/results
    python -m h4d2 adapt --n 25 --seed 42 # Stage 2 model for one n
    python -m h4d2 predict --city Amsterdam --n 25 --input test.parquet --out preds.csv
    python -m h4d2 predict --city Madrid --input test.parquet --out preds.csv

## Method

Stage 1 (Madrid):
* organiser's 60 temporal features, standardised on Madrid, averaged with their 90 m neighbourhood
  (Gaussian weights, sigma 45 m) -> multinomial logistic regression (feature head);
* six 1-D CNNs on the full 42-year series (12 channels: log reflectance, 5 indices, brightness, plus the
  same channels averaged over the neighbourhood). Seeds 42/1/2 plain; seeds 1/2/3 domain-adversarial
  (gradient reversal against unlabelled Amsterdam pixels, weight 0.3).

Stage 2 (n labelled pixels per class):
* every head is re-fitted as a logistic regression penalised toward its Madrid weights
  (lambda 2 for the feature head, 8 for CNN embeddings);
* ensemble = 1/3 feature head + 2/3 mean of the CNN heads;
* probabilities smoothed by label propagation over the neighbour graph (alpha 3, 10 iterations),
  with the labelled pixels clamped to their labels.

Spatial context at predict time uses the test pixels plus, by default, the provided city file as
unlabelled neighbours (--no-context to disable). Stage 2 supports are drawn with
numpy default_rng([seed, n]), n per class, classes in sorted order.

Disclosure: the domain-adversarial CNNs see unlabelled Amsterdam pixel features during Stage 1
(no Amsterdam labels). Hyper-parameters (radius, lambdas, ensemble weight) were chosen on seed 42
supports and confirmed on seed 7.

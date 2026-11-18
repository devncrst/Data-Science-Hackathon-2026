| Result | Macro F1 (mean +/- std) | Prototypes on our features | Method |
| --- | --- | --- | --- |
| Madrid cross-validation, random folds | 0.750 +/- 0.005 | | Stage 1 |
| Madrid cross-validation, 1 km spatial blocks | 0.659 +/- 0.012 | | Stage 1 |
| Amsterdam zero-shot | 0.450 | | Stage 1 only |
| Amsterdam, n = 5 | 0.583 +/- 0.048 | 0.522 +/- 0.054 | blend + align |
| Amsterdam, n = 25 | 0.688 +/- 0.011 | 0.605 +/- 0.024 | stack_et |
| Amsterdam, n = 50 | 0.709 +/- 0.011 | 0.614 +/- 0.011 | stack_et + align |
| Amsterdam, n = 100 | 0.731 +/- 0.009 | 0.620 +/- 0.011 | stack_et |
| Amsterdam, n = 200 | 0.750 +/- 0.004 | 0.623 +/- 0.006 | stack_et |
| Average over sample sizes | 0.692 | 0.597 | |
| Peak | 0.750 | 0.623 | |

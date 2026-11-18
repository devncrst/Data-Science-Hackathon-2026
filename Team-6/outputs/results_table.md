| Result | Macro F1 (mean +/- std) | Baseline prototype | Method |
| --- | --- | --- | --- |
| Madrid cross-validation | 0.750 +/- 0.005 | | Stage 1 |
| Amsterdam zero-shot | 0.450 | | Stage 1 only |
| Amsterdam, n = 5 | 0.601 +/- 0.040 | 0.556 +/- 0.039 | blend + align |
| Amsterdam, n = 25 | 0.689 +/- 0.019 | 0.601 +/- 0.027 | stack_et |
| Amsterdam, n = 50 | 0.710 +/- 0.013 | 0.618 +/- 0.012 | stack_et + align |
| Amsterdam, n = 100 | 0.733 +/- 0.009 | 0.619 +/- 0.011 | stack_et |
| Amsterdam, n = 200 | 0.752 +/- 0.005 | 0.622 +/- 0.007 | stack_et |
| Average over sample sizes | 0.697 | 0.603 | |
| Peak | 0.752 | 0.622 | |

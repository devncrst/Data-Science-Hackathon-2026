# Submission text (v2, Wed 18 Nov 04:30)

## Abstract (slide 1, max 150 words: this is 147 words)

Can a building-age model trained in one city work in another with only a few local labels? We classify 30 m Landsat pixels into four age classes, training on Madrid and adapting to Amsterdam. Our pipeline removes patchy haze with a rule that compares each photo to the pixel's own history, summarises each pixel's 42-year time series into 132 level, change and timing features, and adds the average features of its neighbourhood. A Random Forest learns from Madrid; Amsterdam is calibrated on its own unlabelled pixels, then adapted by averaging a fine-tuned forest, a stacked model and class prototypes, followed by spatial smoothing. Amsterdam macro F1 averages 0.689 across 5 to 200 labels per class (organisers' baseline 0.596), 0.675 at 25 labels. We tested nine further ideas against rules fixed in advance and confirmed the winners on hidden neighbourhoods. The model also knows when it is unsure.

## Written justification (max 300 words: this is 277 words)

Model design. Stage 1 is a Random Forest (300 trees, balanced classes) trained on all Madrid pixels. Each pixel is described by 132 features from its cleaned 1984 to 2025 Landsat series: level, trend and change statistics for six bands and five spectral indices, where change statistics use only observed years so cloudy cities are not biased. We add the average features of neighbouring pixels in 3x3 and 7x7 windows, because buildings of one era cluster. Haze is removed with a temporal rule that compares each photo to the pixel's own typical brightness, which keeps new buildings that a fixed threshold wrongly removed.

Transfer strategy. Amsterdam features are standardised on Amsterdam's own unlabelled pixels, a label-free city calibration (0.689 vs 0.682 with Madrid's scaling). With n labelled pixels per class we average three views: the forest retrained on Madrid plus the weighted Amsterdam pixels, a regularised logistic regression on the Madrid model's probabilities and the scaled features, and class prototypes. Class probabilities are then smoothed over a 5x5 neighbourhood using predictions only.

Interpreting the scores. Amsterdam macro F1 rises from 0.617 (n=5) to 0.675 (n=25), 0.701 (n=50), 0.718 (n=100) and 0.733 (n=200), against 0.544 to 0.615 for the baseline. Gains flatten above 50 labels: the Madrid model supplies most of the knowledge, local labels mainly recalibrate it. Errors fall mostly in neighbouring age classes, because pixel labels average several buildings. Madrid scores 0.733 in random cross-validation and 0.629 when whole 1.5 km areas are held out; we treat the second as the realistic figure. Tests on hidden neighbourhoods, reverse transfer and stress tests support that the method generalises; self-training, boosting and class-mix correction were tested and rejected.

## Numbers for the results slide (00-MAIN-Pipeline-v2, seed 42, 10 support samples per n)

| Setting | Macro F1 (mean +/- std) | Organisers' baseline |
|---|---|---|
| Madrid, random 5x5 CV (optimistic with neighbourhood features) | 0.733 +/- 0.003 | 0.618 |
| Madrid, spatial CV (1.5 km blocks held out, realistic) | 0.629 +/- 0.034 | |
| Amsterdam n = 5 | 0.617 +/- 0.028 | 0.544 |
| Amsterdam n = 25 | 0.675 +/- 0.014 | 0.599 |
| Amsterdam n = 50 | 0.701 +/- 0.010 | 0.608 |
| Amsterdam n = 100 | 0.718 +/- 0.004 | 0.613 |
| Amsterdam n = 200 | 0.733 +/- 0.004 | 0.615 |
| **Amsterdam average of 5** | **0.689** | 0.596 |

Plot: f1_vs_log2n_v2.png. Madrid confusion (spatial CV, smoothed): results/madrid_confusion_proportions_v2_spatial.csv.

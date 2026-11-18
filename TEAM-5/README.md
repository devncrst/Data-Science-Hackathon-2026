# Building age few-shot transfer submission

## Files

- `1_Few_Shot_Transfer.ipynb`: runnable submission notebook with a fixed seed and editable label budgets.
- `5-Improved_Transfer.py`: end-to-end implementation; accepts `--shots`, `--trials`, `--seed`, and `--out-dir`.
- `data/`: Madrid and Amsterdam parquet inputs used by the script.
- `results/stage1_madrid_model.joblib`: exported from the VM Stage 1 fit; contains the Madrid-fitted scaler, class prototypes, within-class precision matrix, and feature mask.
- `results/`: created on run; contains the four-budget F1 summary and F1-versus-log2(n) plot.
- `Building_Age_Classification_Pitch.pptx`: presentation.
- `Written_Justification.md`: <=300-word model and result justification.

## Run

Use Python 3.10 or later. Install the dependencies with `python -m pip install -r requirements.txt`, then open the notebook and run all cells. Or run the script from this folder:

```bash
python 5-Improved_Transfer.py --shots 5,25,100,200 --trials 10 --seed 42 --out-dir results
```

The sample budgets and trial count are configurable. Keep the supplied parquet files in `data/` or update the path constants near the top of the script. The experiment uses Amsterdam labels only to draw the small support set and to score the disjoint query set; its 8-neighbour imagery context is transductive because it uses the full unlabeled scene.

## Submitting the notebook

Submit `1_Few_Shot_Transfer.ipynb` as the notebook deliverable. Include `5-Improved_Transfer.py` and `requirements.txt` so it can run on the grader's environment. Include the two parquet files only if the submission portal permits the dataset size; otherwise submit code and note that the supplied evaluation data must be placed in `data/`. The notebook is a wrapper that runs the same script, so its parameters and outputs stay reproducible.

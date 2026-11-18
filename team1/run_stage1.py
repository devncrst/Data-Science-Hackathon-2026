#!/usr/bin/env python3
"""Madrid-trained Stage 1 inference on engineered NPZ query features.

Expects query102 (N,102), query252 (N,252), optional query_pixel_ids.
This does NOT ingest JPG, PNG, GeoTIFF or raw Parquet files.
"""
import argparse
from pathlib import Path
import numpy as np
from age_hackathon import load_bundle

def run(bundle, input_path, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    with np.load(input_path, allow_pickle=False) as d:
        x102 = d['query102'].copy()
        x252 = d['query252'].copy()
        ids = d['query_pixel_ids'].copy() if 'query_pixel_ids' in d.files else None
    model = load_bundle(bundle)
    predictions, probabilities = model.predict_stage1(x102, x252)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = dict(predictions=predictions, probabilities=probabilities,
                  classes=np.array([1, 2, 3, 4], dtype=int))
    if ids is not None:
        if len(ids) != len(predictions):
            raise ValueError('Pixel IDs must align with predictions')
        fields['query_pixel_ids'] = ids
    with output.open('xb') as handle:
        np.savez_compressed(handle, **fields)
    print(f'SAVED {len(predictions)} Madrid-model predictions to {output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='model_bundle')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.bundle, args.input, args.output)

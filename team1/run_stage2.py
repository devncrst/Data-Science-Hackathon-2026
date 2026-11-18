#!/usr/bin/env python3
"""Runnable Stage 2 CLI. Accepts engineered features, never query labels.

Input NPZ must contain:
  support102     (4*n, 102)
  support252     (4*n, 252), may have NaN for unobserved years
  support_labels (4*n,), n each of 1,2,3,4
  query102       (N, 102)
  query252       (N, 252)
Optional: query_pixel_ids (N,2)
No query labels are read or expected.
"""
import argparse
from pathlib import Path
import numpy as np
from age_hackathon import load_bundle


def run(bundle, input_path, n, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing predictions: {output_path}')
    with np.load(input_path, allow_pickle=False) as data:
        required = {'support102', 'support252', 'support_labels', 'query102', 'query252'}
        if not required.issubset(data.files):
            raise ValueError(f'Input NPZ missing: {sorted(required-set(data.files))}')
        support102 = data['support102'].copy()
        support252 = data['support252'].copy()
        labels = data['support_labels'].copy()
        query102 = data['query102'].copy()
        query252 = data['query252'].copy()
        query_ids = data['query_pixel_ids'].copy() if 'query_pixel_ids' in data.files else None
    predictor = load_bundle(bundle)
    predictor.adapt(support102, support252, labels, n=n)
    preds = predictor.predict(query102, query252)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as handle:
        if query_ids is None:
            np.savez_compressed(handle, predictions=preds, classes=np.arange(1, 5), shots_per_class=np.array([n]))
        else:
            if len(query_ids) != len(preds):
                raise ValueError('query_pixel_ids must align with query predictions')
            np.savez_compressed(handle, predictions=preds, query_pixel_ids=query_ids,
                                classes=np.arange(1, 5), shots_per_class=np.array([n]))
    print(f'SAVED {len(preds):,} predictions (classes 1..4) to {output_path}')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='model_bundle')
    parser.add_argument('--input', required=True, help='NPZ with support/query features and support labels')
    parser.add_argument('--n', type=int, required=True, choices=[5, 25, 50, 100, 200])
    parser.add_argument('--output', required=True, help='new predictions.npz')
    args = parser.parse_args()
    run(args.bundle, args.input, args.n, args.output)

"""Example — the run requires engineered features and permitted target support labels.

NO QUERY LABELS ARE NEEDED.
"""
from age_hackathon import load_bundle

# Your code MUST provide arrays derived from the exact 102+252 schema:
# - support102: (4*n, 102)
# - support252: (4*n, 252)
# - support_labels: (4*n,) containing n of each class 1..4
# - query102: (n_query, 102)
# - query252: (n_query, 252)
#
# predictor = load_bundle('model_bundle')
# predictor.adapt(support102, support252, support_labels, n=200)
# predictions = predictor.predict(query102, query252)
# print(predictions)  # one integer 1..4 per query pixel

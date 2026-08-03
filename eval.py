#!/usr/bin/env python

"""Shadow evaluation: run the trained model over recorded samples and
compare its predictions against what the human actually did.

Usage: uv run python eval.py [num_samples]
"""

import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import create_model, output_activation

def steering_metrics(pred, actual):
    corr = np.corrcoef(pred, actual)[0, 1] if pred.std() > 0 and actual.std() > 0 else float('nan')
    mae = np.abs(pred - actual).mean()
    return corr, mae


if __name__ == '__main__':
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    episodes = np.load("data/episodes.npy") # re-run `python utils.py prepare samples/*` if missing

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load('model_weights.pt', map_location='cpu'))
    model.eval()

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else len(X)

    preds = []
    with torch.no_grad():
        for i in range(0, num_samples, 50):
            x = torch.from_numpy(X[i:i+50]).permute(0, 3, 1, 2).float() / 255.0
            preds.append(output_activation(model(x)).numpy())
    preds = np.concatenate(preds)
    actual = y[:len(preds)]
    episodes = episodes[:len(preds)]

    corr, mae = steering_metrics(preds[:, 0], actual[:, 0])
    print(f'steering over {len(preds)} samples: correlation {corr:.2f}, mean abs error {mae:.2f}')
    print(f'model steering range [{preds[:, 0].min():.2f}, {preds[:, 0].max():.2f}] '
          f'vs human [{actual[:, 0].min():.2f}, {actual[:, 0].max():.2f}]')

    # per-episode breakdown - a single aggregate number hides where the
    # model is weak, and the held-out episode is the number that matters
    print('\nper-episode steering:')
    for ep in np.unique(episodes):
        m = episodes == ep
        corr, mae = steering_metrics(preds[m, 0], actual[m, 0])
        print(f'  {ep:40s} n={m.sum():5d}  corr {corr:+.2f}  mae {mae:.2f}')

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    axes[0].plot(actual[:, 0], 'b', label='human')
    axes[0].plot(preds[:, 0], 'r', alpha=0.7, label='model')
    axes[0].set_ylabel('steering')
    axes[0].legend()
    axes[1].plot(actual[:, 2], 'b')
    axes[1].plot(preds[:, 2], 'r', alpha=0.7)
    axes[1].set_ylabel('a button')
    # disagreement histogram - the high-disagreement tail is where the
    # model needs DAgger data
    axes[2].hist(np.abs(preds[:, 0] - actual[:, 0]), bins=50)
    axes[2].set_ylabel('count')
    axes[2].set_xlabel('|model - human| steering')
    plt.tight_layout()
    plt.savefig('eval.png')
    print('\nwrote eval.png')

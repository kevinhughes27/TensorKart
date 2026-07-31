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

if __name__ == '__main__':
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load('model_weights.pt', map_location='cpu'))
    model.eval()

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    preds = []
    with torch.no_grad():
        for i in range(0, num_samples, 50):
            x = torch.from_numpy(X[i:i+50]).permute(0, 3, 1, 2).float() / 255.0
            preds.append(output_activation(model(x)).numpy())
    preds = np.concatenate(preds)
    actual = y[:len(preds)]

    corr = np.corrcoef(preds[:, 0], actual[:, 0])[0, 1]
    mae = np.abs(preds[:, 0] - actual[:, 0]).mean()
    print(f'steering over {len(preds)} samples: correlation {corr:.2f}, mean abs error {mae:.2f}')
    print(f'model steering range [{preds[:, 0].min():.2f}, {preds[:, 0].max():.2f}] '
          f'vs human [{actual[:, 0].min():.2f}, {actual[:, 0].max():.2f}]')

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(actual[:, 0], 'b', label='human')
    axes[0].plot(preds[:, 0], 'r', alpha=0.7, label='model')
    axes[0].set_ylabel('steering')
    axes[0].legend()
    axes[1].plot(actual[:, 2], 'b')
    axes[1].plot(preds[:, 2], 'r', alpha=0.7)
    axes[1].set_ylabel('a button')
    plt.savefig('eval.png')
    print('wrote eval.png')

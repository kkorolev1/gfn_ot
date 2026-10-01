"""Fit normalized L/R rewards to SUSHI cohorts by maximum likelihood."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp

from envs.sushi.sushi import SUSHI_ITEMS


DATA_URL = 'https://www.kamishima.net/asset/sushi3-2016.zip'
ROOT = Path(__file__).resolve().parents[2]


def load_sushi(archive):
    """Rows are aligned by position, NOT by user ID or the leading order-file 0."""
    with ZipFile(archive) as data:
        orders = data.read('sushi3-2016/sushi3a.5000.10.order').decode('utf-8')
        header, body = orders.split('\n', 1)
        if header.split() != ['10', '1']:
            raise ValueError('Expected SUSHI set A, with ten shared items')
        rows = np.loadtxt(io.StringIO(body), dtype=np.int32, ndmin=2)
        users = np.loadtxt(io.BytesIO(data.read('sushi3-2016/sushi3.udata')), dtype=np.int32, ndmin=2)
    if rows.shape[1] != 12 or np.any(rows[:, 0] != 0) or np.any(rows[:, 1] != 10):
        raise ValueError('Every SUSHI-A row must contain a complete ten-item ranking')
    rankings = rows[:, 2:]
    if not np.all(np.sort(rankings, axis=1) == np.arange(10)):
        raise ValueError('Invalid permutation in the ranking file')
    if users.shape != (len(rankings), 11) or len(np.unique(users[:, 0])) != len(users):
        raise ValueError('User metadata must have one distinct, aligned user per ranking')
    if np.any(~np.isin(users[:, 1], [0, 1])) or np.any(~np.isin(users[:, 2], np.arange(6))):
        raise ValueError('Unexpected gender or age code in user metadata')
    return rankings, users


def split_groups(rankings, users, split, age_cutoff):
    if split == 'gender':
        mask = users[:, 1] == 0
        labels = ('male', 'female')
    elif split == 'age':
        if age_cutoff not in (20, 30, 40, 50, 60):
            raise ValueError('age_cutoff must coincide with a dataset age-bin boundary')
        mask = users[:, 2] < age_cutoff // 10 - 1
        labels = (f'under_{age_cutoff}', f'{age_cutoff}_and_over')
    else:
        raise ValueError(f'Unknown cohort split: {split}')
    if not np.any(mask) or np.all(mask):
        raise ValueError('Both cohorts must contain rankings')
    return rankings[mask], rankings[~mask], labels


def fit_plackett_luce(rankings):
    rankings = np.asarray(rankings)
    if rankings.ndim != 2 or min(rankings.shape) < 2:
        raise ValueError('At least two full rankings of two or more items are required')
    count, n = rankings.shape
    if not np.issubdtype(rankings.dtype, np.integer) or not np.all(np.sort(rankings, axis=1) == np.arange(n)):
        raise ValueError('Rankings must be permutations of 0, ..., n-1')
    positions = np.argsort(rankings, axis=-1)
    wins = np.any(positions[:, :, None] < positions[:, None, :], axis=0)
    if connected_components(wins, directed=True, connection='strong', return_labels=False) != 1:
        raise ValueError('Comparison graph is not strongly connected; no finite unregularized MLE')

    def objective(free_parameters):
        # Fix one log-worth to remove the additive non-identifiability.
        log_worth = np.r_[free_parameters, 0.0]
        ordered = log_worth[rankings]
        log_denominator = np.logaddexp.accumulate(ordered[:, ::-1], axis=-1)[:, ::-1]
        nll = np.mean(np.sum(log_denominator - ordered, axis=-1))
        log_inverse_sums = np.logaddexp.accumulate(-log_denominator, axis=-1)
        ordered_gradient = np.exp(ordered + log_inverse_sums) - 1
        gradient = np.bincount(rankings.ravel(), weights=ordered_gradient.ravel(), minlength=n) / count
        return nll, gradient[:-1]

    result = minimize(objective, np.zeros(n - 1), jac=True, method='L-BFGS-B',
                      options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 2000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'Plackett-Luce fit did not converge: {result.message}')
    log_worth = np.r_[result.x, 0.0]
    log_worth -= logsumexp(log_worth)
    report = {'num_rankings': count, 'mean_nll': float(result.fun),
              'uniform_mean_nll': math.lgamma(n + 1), 'iterations': int(result.nit),
              'gradient_max_abs': float(np.max(np.abs(result.jac))),
              'converged': bool(result.success), 'worth': np.exp(log_worth).tolist()}
    return log_worth, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / '.cache/sushi/sushi3-2016.zip')
    parser.add_argument('--split', choices=('gender', 'age'), default='gender')
    parser.add_argument('--age-cutoff', type=int, choices=(20, 30, 40, 50, 60), default=40)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.archive.exists():
        print(f'Downloading {DATA_URL} to {args.archive}')
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_bytes(content)
    rankings, users = load_sushi(args.archive)
    left, right, labels = split_groups(rankings, users, args.split, args.age_cutoff)
    left_log_worth, left_report = fit_plackett_luce(left)
    right_log_worth, right_report = fit_plackett_luce(right)
    suffix = args.split if args.split == 'gender' else f'age{args.age_cutoff}'
    output = args.output or ROOT / f'envs/sushi/checkpoint/plackett_luce_{suffix}.npz'
    if output.suffix != '.npz':
        parser.error('--output must end in .npz')
    metadata = {
        'format_version': 1, 'model': 'Plackett-Luce', 'dataset_url': DATA_URL,
        'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'order_file': 'sushi3a.5000.10.order', 'ordering': 'most_preferred_first',
        'split': args.split, 'item_names': list(SUSHI_ITEMS),
        'left': {'group': labels[0], **left_report},
        'right': {'group': labels[1], **right_report},
        'logZ_left': 0.0, 'logZ_right': 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, left_log_worth=left_log_worth,
                        right_log_worth=right_log_worth, metadata=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for side in ('left', 'right'):
        fit = metadata[side]
        print(f"{side}: {fit['group']}, n={fit['num_rankings']}, NLL={fit['mean_nll']:.6f}, "
              f"uniform NLL={fit['uniform_mean_nll']:.6f}, converged={fit['converged']}")
    print(f'Saved normalized L/R: {output}')


if __name__ == '__main__':
    main()

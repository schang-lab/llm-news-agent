"""Exact finite-experiment Bernoulli TVD expectation, using only NumPy.

Enumerate nonempty frame subsets, then convolve each frame's possible shares.
Expected TVD is the sum of these marginal absolute deviations; no Monte Carlo
approximation or full joint distribution across frames is needed here.
"""

from math import lcm

import numpy as np


def reference(counts, outputs, query_keys):
    """Return observed TVD, expected reference TVD and their difference per draw.

    Counts have shape (annotation draw, answer, frame), out of ten input articles.
    Outputs use the same layout and contain answer frame scores.
    Reference frames are included independently at their source mention rates,
    conditional on a nonempty pseudo-answer; the five queries are weighted equally.
    """
    rates = np.asarray(counts, dtype=float) / 10
    mentioned = np.asarray(outputs) > 0
    queries = np.array([int(q.split("_")[-1]) - 1 for q in query_keys])
    n_draws, _, n_frames = rates.shape
    result = np.full((n_draws, 3), np.nan)
    # Use the same nonempty input/answer pairs for observed and reference TVD.
    valid = (rates.sum(2) > 0) & (mentioned.sum(2) > 0)
    subsets = (np.arange(1, 2 ** n_frames)[:, None] >> np.arange(n_frames)) & 1
    sizes = subsets.sum(1)

    # Draws with the same valid answers share query weights and the share grid.
    patterns, membership = np.unique(valid, axis=0, return_inverse=True)
    for group, keep in enumerate(patterns):
        query_counts = np.bincount(queries[keep], minlength=5)
        if np.any(query_counts == 0):
            continue
        # Give each of five queries equal weight, then average its valid answers.
        denominators = 5 * query_counts[queries[keep]]
        weights = 1 / denominators
        # Every weighted 1/k share is an integer position on this common share grid.
        share_grid = lcm(*(int(d * k) for d in denominators for k in range(1, n_frames + 1)))
        selected = np.flatnonzero(membership == group)
        # Small batches bound memory when a missing answer increases the grid size.
        for start in range(0, len(selected), 64):
            indices = selected[start:start + 64]
            input_rates = rates[indices][:, keep]
            output_mentions = mentioned[indices][:, keep]
            input_profile = (
                weights[None, :, None] * input_rates / input_rates.sum(2, keepdims=True)
            ).sum(1)
            output_profile = (
                weights[None, :, None] * output_mentions / output_mentions.sum(2, keepdims=True)
            ).sum(1)
            observed = .5 * np.abs(output_profile - input_profile).sum(1)
            subset_probabilities = np.where(
                subsets[None, None], input_rates[:, :, None], 1 - input_rates[:, :, None],
            ).prod(3)
            # Condition on at least one frame; each pseudo-answer sums to one.
            subset_probabilities /= subset_probabilities.sum(2, keepdims=True)
            expectation = np.zeros(len(indices))
            for frame in range(n_frames):
                # atoms[..., k]: this frame receives share 1/k, or zero when k=0.
                atoms = np.zeros((len(indices), len(denominators), n_frames + 1))
                atoms[:, :, 0] = subset_probabilities[:, :, subsets[:, frame] == 0].sum(2)
                for k in range(1, n_frames + 1):
                    included = (subsets[:, frame] == 1) & (sizes == k)
                    atoms[:, :, k] = subset_probabilities[:, :, included].sum(2)
                distribution = np.ones((len(indices), 1))
                for run, denominator in enumerate(denominators):
                    active = np.flatnonzero(np.any(atoms[:, run] > 0, axis=0))
                    steps = [
                        0 if k == 0 else share_grid // (int(denominator) * k) for k in active
                    ]
                    updated = np.zeros((len(indices), distribution.shape[1] + max(steps)))
                    for k, step in zip(active, steps):
                        updated[:, step:step + distribution.shape[1]] += (
                            atoms[:, run, k, None] * distribution
                        )
                    distribution = updated
                shares = np.arange(distribution.shape[1]) / share_grid
                expectation += .5 * (
                    distribution * np.abs(shares[None] - input_profile[:, frame, None])
                ).sum(1)
            result[indices] = np.column_stack([observed, expectation, observed - expectation])
    return result

"""Event estimates, bootstrap intervals and frame-characteristic comparisons.

Score arrays start with an annotation-draw axis (length one for original scores).
Article scores then have text/frame axes; retrieved scores have run/article/frame axes.
"""

from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

from config import CONFIG
from .bernoulli import reference
from .tables import score_matrix

MODELS = CONFIG["models"]
BASELINES = CONFIG["baselines"]


def mean(x, axis):
    """Average available values, leaving empty curve bins as NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def normalize(x):
    """Convert the last axis to shares, returning NaN for zero-total profiles."""
    total = x.sum(axis=-1, keepdims=True)
    # Zero-total profiles have undefined shares; preserve them as NaN.
    return np.divide(x, total, out=np.full(x.shape, np.nan, dtype=float), where=total > 0)


def query_average(x, keys):
    """Average replicates within queries, then queries equally; runs are axis 1."""
    # This prevents a query with more successful runs from receiving more weight.
    return mean(np.stack([mean(x[:, keys == q], 1) for q in sorted(set(keys))]), 0)


def variance(scores):
    """Variance of normalized frame scores within each article or answer."""
    shares = normalize(scores.astype(float))
    # Population variance across frames within one text, not across events.
    return np.var(shares, axis=-1)


def interval(values, indices):
    """Return the event mean and an unadjusted 95% percentile bootstrap interval."""
    values = np.asarray(values)
    # Each index row resamples whole events, preserving within-event dependence.
    boot = mean(values[indices], 1)
    n = np.isfinite(values).sum()
    low, high = np.nanquantile(boot, [.025, .975]) if n >= 2 else (np.nan, np.nan)
    return dict(
        mean=float(mean(values, 0)), ci_low=float(low), ci_high=float(high),
        events=int(n), significant=bool(low > 0 or high < 0),
    )


def correlation(x, y):
    """Pearson r along the last axis, including bootstrap-repeated events."""
    x, y = np.broadcast_arrays(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    x, y = x - mean(x, -1)[..., None], y - mean(y, -1)[..., None]
    numerator = np.nansum(x * y, axis=-1)
    denominator = np.sqrt(np.nansum(x * x, axis=-1) * np.nansum(y * y, axis=-1))
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0,
    )


def retrieval_metrics(corpus_scores, selected_scores, query_ids):
    """Return corpus-relative event metrics and frame shifts, weighting queries equally."""
    corpus_any = (corpus_scores > 0).mean(1)
    corpus_central = (corpus_scores == 2).mean(1)
    any_rate = (selected_scores > 0).mean(2)
    central_rate = (selected_scores == 2).mean(2)
    mention = query_average(any_rate, query_ids)
    retrieved_profile = query_average(normalize(any_rate), query_ids)
    observed = .5 * np.abs(retrieved_profile - normalize(corpus_any)).sum(-1)
    metrics = {
        "any_shift_pp": 100 * (mention - corpus_any).mean(-1),
        "central_shift_pp": 100 * (query_average(central_rate, query_ids) - corpus_central).mean(-1),
        "mention_tvd": observed,
        "variance_change": (
            query_average(mean(variance(selected_scores), 2), query_ids)
            - mean(variance(corpus_scores), 1)
        ),
    }
    frame_shift = query_average(selected_scores.mean(2), query_ids) - corpus_scores.mean(1)
    return metrics, frame_shift


def synthesis_metrics(input_scores, answer_scores, query_ids, bernoulli=True):
    """Return paired answer-input metrics and per-frame score shifts for one event.

    Input scores have axes (draw, run, article, frame); answers omit the article axis.
    """
    mentioned = (input_scores > 0).sum(2)
    central = (input_scores == 2).sum(2)
    changes = {
        "any_shift_pp": 100 * ((answer_scores > 0) - mentioned / 10).mean(-1),
        "central_shift_pp": 100 * ((answer_scores == 2) - central / 10).mean(-1),
        "variance_change": variance(answer_scores) - mean(variance(input_scores), 2),
    }
    metrics = {name: query_average(values, query_ids) for name, values in changes.items()}
    if bernoulli:
        # The input-rate reference is recomputed for every annotation draw.
        expected = reference(mentioned, answer_scores, query_ids)
        for index, name in enumerate(["observed", "bernoulli_reference", "excess"]):
            metrics[name] = expected[:, index]
    frame_shift = query_average(answer_scores - input_scores.mean(2), query_ids)
    return metrics, frame_shift


def transfer_curves(input_scores, answer_scores, query_ids):
    """Answer mention/central rates at each input-article count, within one event."""
    curves = {}
    conditions = [("mentioned", (input_scores > 0).sum(2)),
                  ("central", (input_scores == 2).sum(2))]
    for axis, condition in conditions:
        outcomes = [("mentioned", answer_scores > 0), ("central", answer_scores == 2)]
        for level in range(11):
            mask = condition == level
            for outcome, present in outcomes:
                per_query = []
                for query in sorted(set(query_ids)):
                    take = query_ids == query
                    count = mask[:, take].sum(1)
                    # Average replicates within each frame/bin, then average frames.
                    by_frame = np.divide(
                        (present[:, take] * mask[:, take]).sum(1), count,
                        out=np.full(count.shape, np.nan), where=count > 0,
                    )
                    per_query.append(mean(by_frame, -1))
                curves[f"{axis}:{outcome}:{level}"] = mean(np.stack(per_query), 0)
    return curves


def selected_scores(corpus_scores, article_ids, selected_article_ids):
    """Select article scores in retrieval order, adding a run axis."""
    positions = {article: index for index, article in enumerate(article_ids)}
    selected = np.array([[positions[article] for article in ids] for ids in selected_article_ids])
    return corpus_scores[:, selected]


def calculate_events(articles, frames, article_scores, answer_scores, retrieval, answers, queries,
                     labels=None, bernoulli=True, curves=True):
    """Return event estimates, frame shifts and score-derived slants for neutral queries.

    Each estimate keeps its annotation-draw and event axes for later event bootstraps.
    Optional labels replace the original scores through their shared score indices.
    """
    events = sorted(frames.event_name.unique())
    draws = 1 if labels is None else len(labels)
    values = defaultdict(lambda: np.full((draws, len(events)), np.nan))
    shifts = {stage: {model: [] for model in MODELS} for stage in ["retrieval", "synthesis"]}
    slants = []

    def save(stage, model, metric, event, value, role="", group=""):
        """Store all annotation draws for one metric and event."""
        values[stage, model, metric, role, group][:, event] = value

    for event_index, event in enumerate(events):
        frame_ids = frames[frames.event_name.eq(event)].frame_id.tolist()
        event_articles = articles[articles.event_name.eq(event)]
        article_ids = event_articles.article_id.tolist()
        corpus_scores = score_matrix(article_scores, "article_id", article_ids, frame_ids, labels)
        lean = event_articles.slant_code.to_numpy()
        central_count = (corpus_scores == 2).sum(1)
        # Score-derived slant changes with the perturbed article annotations.
        slants.append(np.divide(
            ((corpus_scores == 2) * lean[None, :, None]).sum(1), central_count,
            out=np.full(central_count.shape, np.nan), where=central_count > 0,
        ))

        runs = retrieval[
            retrieval.event_name.eq(event) & retrieval.query_id.str.startswith("neutral")
        ]
        by_method = {}
        for method, group in runs.groupby("retriever"):
            selected = selected_scores(corpus_scores, article_ids, group.selected_article_ids)
            metrics, frame_shift = retrieval_metrics(
                corpus_scores, selected, group.query_id.to_numpy(),
            )
            by_method[method] = metrics
            for name, value in metrics.items():
                save("retrieval_corpus", method, name, event_index, value)
            if method in MODELS:
                shifts["retrieval"][method].append(frame_shift)
        for method in BASELINES + MODELS:
            for name in ["any_shift_pp", "central_shift_pp", "mention_tvd"]:
                difference = by_method[method][name] - by_method["random"][name]
                save("retrieval_random", method, name, event_index, difference)

        event_answers = answers[answers.event_name.eq(event)]
        for model in MODELS:
            group = event_answers[
                event_answers.model.eq(model) & event_answers.query_id.str.startswith("neutral")
            ]
            keys = group.query_id.to_numpy()
            inputs = selected_scores(corpus_scores, article_ids, group.selected_article_ids)
            outputs = score_matrix(
                answer_scores, "synthesis_experiment_id", group.synthesis_experiment_id,
                frame_ids, labels,
            )
            metrics, frame_shift = synthesis_metrics(inputs, outputs, keys, bernoulli)
            for name, value in metrics.items():
                save("synthesis", model, name, event_index, value)
            shifts["synthesis"][model].append(frame_shift)
            if curves:
                for name, value in transfer_curves(inputs, outputs, keys).items():
                    save("curve", model, name, event_index, value)

    frame_shifts = {
        stage: {model: np.concatenate(parts, axis=1) for model, parts in methods.items()}
        for stage, methods in shifts.items()
    }
    return dict(
        values=dict(values), frame_shifts=frame_shifts, slants=np.concatenate(slants, axis=1),
    )


def summarize_events(result, bootstrap):
    """Summarize event estimates with bootstrap confidence intervals."""
    rows = []
    for (stage, model, metric, role, group), estimates in result["values"].items():
        description = dict(stage=stage, model=model, metric=metric, role=role, group=group)
        # Main confidence intervals use original annotations and resample events.
        rows.append({**description, **interval(estimates[0], bootstrap)})
    return pd.DataFrame(rows)


def analyze_characteristics(result, frames, bootstrap):
    """Return characteristic statistics with event-bootstrap CIs and per-frame shifts."""
    events = sorted(frames.event_name.unique())
    frame_event = np.array([events.index(e) for e in frames.event_name])
    # An integer weight repeats all frames belonging to a resampled event.
    weights = np.array([np.bincount(i, minlength=len(events)) for i in bootstrap])[:, frame_event]
    rows, frame_rows = [], []
    families = {
        "entman_functions": [
            "define problems", "diagnose causes", "make moral judgments", "suggest remedies",
        ],
        "sv_frame": ["conflict", "human interest", "economic", "morality", "responsibility"],
        "card_categories": [
            "political", "legal", "economic", "human-wellbeing", "values", "cultural",
            "public-opinion",
        ],
    }
    columns = {"entman_functions": "function", "card_categories": "category", "sv_frame": "type"}
    for stage, methods in result["frame_shifts"].items():
        for model, shifts in methods.items():
            # Draw 0 contains the original scores; columns are event-specific frames.
            y = shifts[0]
            frame_rows.extend(
                dict(event=event, frame=frame_id, stage=stage, model=model, shift=float(value))
                for event, frame_id, value in zip(frames.event_name, frames.frame_id, y)
            )
            for characteristic in ["political_slant_score", "valence"]:
                column = "slant" if characteristic == "political_slant_score" else "valence_score"
                x = frames[column].to_numpy(float)
                original = float(correlation(x, y))
                boot = []
                for sample in bootstrap:
                    # Repeated events repeat all their frames in the correlation.
                    indices = np.concatenate([np.flatnonzero(frame_event == e) for e in sample])
                    boot.append(float(correlation(x[indices], y[indices])))
                low, high = np.nanquantile(boot, [.025, .975])
                rows.append(dict(
                    stage=stage, model=model, family="correlation", characteristic=characteristic,
                    mean=original, ci_low=low, ci_high=high, n_frames=len(y),
                ))
            for family, names in families.items():
                for name in names:
                    column = frames[columns[family]]
                    if family == "sv_frame":
                        mask = column.eq(name).to_numpy()
                    else:
                        mask = column.map(lambda labels: name in labels).to_numpy()
                    original = y[mask].mean() - y[~mask].mean()
                    # Bootstrap group sizes vary because an event brings all its frames.
                    left_n, right_n = weights[:, mask].sum(1), weights[:, ~mask].sum(1)
                    # If a resample has no frames in a group, its contrast is undefined.
                    left = np.divide(
                        weights[:, mask] @ y[mask], left_n,
                        out=np.full(len(weights), np.nan), where=left_n > 0,
                    )
                    right = np.divide(
                        weights[:, ~mask] @ y[~mask], right_n,
                        out=np.full(len(weights), np.nan), where=right_n > 0,
                    )
                    boot = left - right
                    low, high = np.nanquantile(boot, [.025, .975])
                    rows.append(dict(
                        stage=stage, model=model, family=family, characteristic=name,
                        mean=original, ci_low=low, ci_high=high, n_frames=int(mask.sum()),
                    ))
    return pd.DataFrame(rows), pd.DataFrame(frame_rows)

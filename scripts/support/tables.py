"""Load analysis tables and match results to their query labels."""

import json

import numpy as np
import pandas as pd

from config import CONFIG
from .common import read_csv


def read_queries():
    """Load the configured query CSVs with their event and query identifiers."""
    return pd.concat(
        [read_csv(path) for path in CONFIG["query_files"]], ignore_index=True,
    ).fillna("")


def read_frames(path):
    """Load final frames in event order, splitting labels and coding valence numerically."""
    frames = (
        read_csv(path)
        .sort_values("event_name", kind="stable")
        .reset_index(drop=True)
    )
    # Function and category can each contain several human-assigned labels.
    for column in ["function", "category"]:
        frames[column] = frames[column].map(lambda value: value.split("; ") if value else [])
    frames["valence_score"] = frames.valence.map({"negative": -1, "neutral": 0, "positive": 1})
    return frames


def read_retrieval_results(paths, query_labels):
    """Load successful retrievals, decode article ID lists and attach query labels."""
    columns = [
        "retrieval_experiment_id", "event_name", "query", "replicate", "retriever",
        "selected_article_ids", "status",
    ]
    runs = pd.concat([read_csv(path, usecols=columns) for path in paths], ignore_index=True)
    runs = runs[runs.status.eq("complete")].copy()
    runs["selected_article_ids"] = runs.selected_article_ids.map(json.loads)
    return runs.merge(
        query_labels, on=["event_name", "query"], how="left", sort=False, validate="many_to_one",
    )


def read_synthesis_results(paths, query_labels):
    """Load successful answers, decode their input ID lists and attach query labels."""
    columns = [
        "synthesis_experiment_id", "event_name", "query", "retrieval_experiment_id", "model",
        "selected_article_ids", "output", "status",
    ]
    answers = pd.concat([read_csv(path, usecols=columns) for path in paths], ignore_index=True)
    answers = answers[answers.status.eq("complete")].copy()
    answers["selected_article_ids"] = answers.selected_article_ids.map(json.loads)
    return answers.merge(
        query_labels, on=["event_name", "query"], how="left", sort=False, validate="many_to_one",
    )


def read_synthesis_annotations(paths):
    """Combine answer-frame scores without loading the repeated answer text."""
    columns = ["synthesis_experiment_id", "frame_id", "score"]
    return pd.concat([read_csv(path, usecols=columns) for path in paths], ignore_index=True)


def score_matrix(scores, id_column, item_ids, frame_ids, labels=None):
    """Arrange scores as (annotation draw, text, frame), in the requested ID order.

    Original scores have one draw. Optional labels are indexed through score_index
    and have shape (draw, score row) before being arranged into the output matrix.
    """
    column = "score" if labels is None else "score_index"
    matrix = scores.pivot(index=id_column, columns="frame_id", values=column)
    matrix = matrix.loc[item_ids, frame_ids]
    if labels is None:
        return matrix.to_numpy(dtype=np.uint8)[None, :]
    return labels[:, matrix.to_numpy(dtype=int)]

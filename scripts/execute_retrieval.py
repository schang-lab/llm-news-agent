"""Run retrieval experiments, reusing cached responses and embeddings when available."""

import argparse
import hashlib
import json

import numpy as np

from config import CONFIG
from support.common import Cache, digest, hits, read_csv, read_recorded, request, save_csv
from support.prompts import messages, opaque_ids
from support.retrievers import Embeddings, bm25, dense, uniform_sets


def paper_random_sets(articles):
    """Recreate the 25 sampled ten-article sets for each event."""
    sets = {}
    for event, pool in articles.groupby("event_name", sort=False):
        identity = int(hashlib.sha256(event.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG["random_seed"], identity]))
        sets[event] = uniform_sets(
            rng, len(pool), 5 * CONFIG["retrieval_replicates"], CONFIG["articles_per_set"],
        )
    return sets


def recorded_random_slots():
    """Map saved experiment IDs to their query/replicate position among the 25 sets."""
    queries = read_recorded(CONFIG["query_files"], ["event_name", "query", "query_id"])
    recorded = read_recorded(CONFIG["retrieval_experiment_files"], [
        "retrieval_experiment_id", "event_name", "query", "replicate", "retriever",
    ])
    recorded = recorded[recorded.retriever.eq("random")].merge(
        queries, on=["event_name", "query"],
    )
    slots = {}
    for row in recorded.itertuples():
        query_index = int(row.query_id.split("_")[-1]) - 1
        slot = query_index * CONFIG["retrieval_replicates"] + row.replicate
        slots[row.retrieval_experiment_id] = slot
    return slots


def random_retrieval(experiment, pool, saved_sets, saved_slots):
    """Return sampled article IDs, preserving study draws or seeding a new experiment."""
    # Preserve the paper's sampling stream for its original experiments.
    if experiment.retrieval_experiment_id in saved_slots:
        slot = saved_slots[experiment.retrieval_experiment_id]
        indices = saved_sets[experiment.event_name][slot]
    else:
        identity = dict(
            event=experiment.event_name, query=experiment.query,
            replicate=int(experiment.replicate),
        )
        seed = int(digest(identity)[:16], 16)
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG["random_seed"], seed]))
        indices = uniform_sets(rng, len(pool), 1, CONFIG["articles_per_set"])[0]
    return pool.iloc[indices].index.tolist()


def execute_retrieval(experiments_path, output_path, corpus_path):
    """Write selected article IDs and status for each LLM or baseline retrieval experiment."""
    experiments = read_csv(experiments_path)
    articles = read_csv(corpus_path).set_index("article_id")
    cache = Cache() if experiments.retriever.isin(CONFIG["models"]).any() else None
    embeddings = Embeddings() if experiments.retriever.str.startswith("dense").any() else None
    recorded = read_recorded(
        CONFIG["retrieval_result_files"], ["retrieval_experiment_id", "status"],
    )
    failures = set(recorded.loc[recorded.status.ne("complete"), "retrieval_experiment_id"])
    random_sets, random_slots = {}, {}
    if experiments.retriever.eq("random").any():
        random_sets = paper_random_sets(articles)
        random_slots = recorded_random_slots()

    rows = []
    for experiment in experiments.itertuples(index=False):
        selected_ids, output, status = [], "", "complete"
        if experiment.retrieval_experiment_id in failures:
            # Retain recorded study exclusions during exact replay.
            status = "failed"
        else:
            pool = articles.loc[json.loads(experiment.shuffled_articles)]
            method = experiment.retriever
            if method in CONFIG["models"]:
                response = cache.record(request(
                    "retrieval", method, messages(experiment.system_prompt, experiment.query),
                    experiment.replicate, CONFIG["retrieval_settings"][method],
                ))
                output = response["raw_output"]
                # Models select temporary position labels, not corpus article IDs.
                article_ids = dict(zip(opaque_ids(len(pool)), pool.index))
                selected_ids = [article_ids[label] for label in json.loads(output)["article_ids"]]
                if isinstance(response["output"], list):
                    # Older cache records also store the decoded corpus IDs.
                    assert selected_ids == response["output"]
            elif method.startswith("bm25"):
                selected_ids = bm25(pool, experiment.query, full=method == "bm25_full")
            elif method.startswith("dense"):
                selected_ids = dense(
                    pool, experiment.query, embeddings, full=method == "dense_first_chunk",
                )
            elif method == "random":
                selected_ids = random_retrieval(experiment, pool, random_sets, random_slots)
            else:
                raise ValueError(f"Unknown retriever: {method}")
            if method not in CONFIG["models"]:
                output = json.dumps(selected_ids)
            assert len(selected_ids) == CONFIG["articles_per_set"]
            assert len(set(selected_ids)) == len(selected_ids)
        rows.append({
            **experiment._asdict(), "output": output,
            "selected_article_ids": json.dumps(selected_ids), "status": status,
        })
    save_csv(rows, output_path)
    hits(str(output_path), cache, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", required=True, help="Input experiments: .csv or .csv.gz")
    parser.add_argument("-o", "--output", required=True, help="Output results: .csv or .csv.gz")
    parser.add_argument("--corpus", required=True, help="Article pool: .csv or .csv.gz")
    args = parser.parse_args()
    execute_retrieval(args.experiments, args.output, args.corpus)

"""Build one synthesis request per successful LLM retrieval."""

import argparse
import json

import pandas as pd

from config import CONFIG
from support.common import csv_hash, digest, preserve_order, read_csv, read_recorded, save_csv
from support.prompts import synthesis


def setup_synthesis(retrieval_path, corpus_path, output_path):
    """Write a synthesis request for each successful LLM retrieval and its ordered inputs."""
    retrieval = read_csv(retrieval_path)
    articles = read_csv(corpus_path).set_index("article_id")
    recorded = read_recorded(CONFIG["synthesis_experiment_files"], [
        "synthesis_experiment_id", "event_name", "query", "retrieval_experiment_id",
        "model", "replicate", "selected_article_ids",
    ])
    lookup = recorded.set_index("retrieval_experiment_id")
    corpus_hash = csv_hash(corpus_path)
    successful = retrieval[
        retrieval.status.eq("complete") & retrieval.retriever.isin(CONFIG["models"])
    ]
    rows = []
    for result in successful.itertuples():
        # Pass the selected articles to the same model, in retrieved order.
        selected_ids = json.loads(result.selected_article_ids)
        system = synthesis(articles, selected_ids, result.query)[0]["content"]
        original = None
        replicate = CONFIG["synthesis_replicate"]
        if result.retrieval_experiment_id in lookup.index:
            original = lookup.loc[result.retrieval_experiment_id]
            replicate = int(original.replicate)
        row = dict(
            event_name=result.event_name, query=result.query,
            retrieval_experiment_id=result.retrieval_experiment_id, model=result.retriever,
            replicate=replicate, selected_article_ids=result.selected_article_ids,
            system_prompt=system,
        )
        experiment_id = digest(dict(stage="synthesis", corpus_sha256=corpus_hash, **row))
        # Public prompts have redacted bodies; match the original corpus and request metadata.
        if corpus_hash == CONFIG["corpus_sha256"] and original is not None:
            same_request = (
                original.event_name == result.event_name and original.query == result.query
                and original.model == result.retriever
                and json.loads(original.selected_article_ids) == selected_ids
            )
            if same_request:
                experiment_id = original.synthesis_experiment_id
        rows.append(dict(synthesis_experiment_id=experiment_id, **row))
    table = preserve_order(pd.DataFrame(rows), "synthesis_experiment_id", recorded)
    save_csv(table, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    setup_synthesis(args.retrieval_results, args.corpus, args.output)

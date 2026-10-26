"""Build retrieval requests from a corpus and a query CSV."""

import argparse
import json

import pandas as pd

from config import CONFIG
from support.common import csv_hash, digest, preserve_order, read_csv, read_recorded, save_csv
from support.prompts import retrieval


def setup_retrieval(corpus_path, queries_path, output_path, retrievers=None):
    """Write query/retriever/replicate requests, retaining matching study IDs and order."""
    articles = read_csv(corpus_path).set_index("article_id")
    queries = read_csv(queries_path)
    recorded = read_recorded(CONFIG["retrieval_experiment_files"], [
        "retrieval_experiment_id", "event_name", "query", "replicate", "retriever",
        "shuffled_articles", "system_prompt",
    ])
    lookup = recorded.set_index(["event_name", "query", "retriever", "replicate"])
    corpus_hash = csv_hash(corpus_path)
    retrievers = retrievers or CONFIG["baselines"] + ["random"] + CONFIG["models"]
    rows = []
    for query in queries.itertuples():
        pool = articles[articles.event_name.eq(query.event_name)]
        for method in retrievers:
            is_llm = method in CONFIG["models"]
            # Deterministic baselines need only one run per query.
            repeated = is_llm or method == "random"
            replicates = range(CONFIG["retrieval_replicates"]) if repeated else [0]
            for replicate in replicates:
                presented = pool.sample(frac=1, random_state=replicate) if is_llm else pool
                system = retrieval(pool, query.query, replicate)[0]["content"] if is_llm else ""
                row = dict(
                    event_name=query.event_name, query=query.query, replicate=replicate,
                    retriever=method, shuffled_articles=json.dumps(presented.index.tolist()),
                    system_prompt=system,
                )
                key = (query.event_name, query.query, method, replicate)
                experiment_id = digest(dict(stage="retrieval", corpus_sha256=corpus_hash, **row))
                # Keep historical IDs when the corpus and request are unchanged.
                if corpus_hash == CONFIG["corpus_sha256"] and key in lookup.index:
                    original = lookup.loc[key]
                    same_order = json.loads(original.shuffled_articles) == presented.index.tolist()
                    if original.system_prompt == system and same_order:
                        experiment_id = original.retrieval_experiment_id
                rows.append(dict(retrieval_experiment_id=experiment_id, **row))
    table = preserve_order(pd.DataFrame(rows), "retrieval_experiment_id", recorded)
    save_csv(table, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--retrievers", nargs="+", choices=CONFIG["baselines"] + ["random"] + CONFIG["models"],
    )
    args = parser.parse_args()
    setup_retrieval(args.corpus, args.queries, args.output, args.retrievers)

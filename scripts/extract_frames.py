"""Extract an article's frame and supporting information, using the cache when available."""

import argparse
import json

import pandas as pd

from config import CONFIG
from support.common import Cache, digest, hits, read_csv, save_csv
from support.prompts import extraction_request


def extract_frames(corpus_path, output_path):
    """Write article frames and supporting points, skipping bodies below the word minimum."""
    cache = Cache("cache/frame_development.jsonl.gz")
    rows = []
    # Saved extraction order preserves grouping prompts and assignment batches.
    positions = {key: index for index, key in enumerate(cache.records)}
    for article in read_csv(corpus_path).itertuples():
        if len(article.body.split()) < CONFIG["min_extraction_words"]:
            continue
        req = extraction_request(article)
        response = cache.get(req)
        positions.setdefault(digest(req), len(positions))
        rows.append(dict(
            _position=positions[digest(req)], article_id=article.article_id,
            event_name=article.event_name, extracted_frame=response["media_frame"],
            salient_information=json.dumps(response["key_points"]),
        ))
    table = pd.DataFrame(rows).sort_values("_position").drop(columns="_position")
    save_csv(table, output_path)
    hits("extraction", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    extract_frames(args.corpus, args.output)

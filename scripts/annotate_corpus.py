"""Score each article against its event's final frames."""

import argparse

from config import CONFIG
from support.common import Cache, frames, hits, read_csv, request, save_csv
from support.prompts import annotation


def annotate_corpus(corpus_path, frames_path, output_path):
    """Score all event frames per article and write one row per article-frame pair."""
    cache = Cache()
    inventories = frames(frames_path)
    rows = []
    for article in read_csv(corpus_path).itertuples():
        inventory = inventories[article.event_name]
        # Exact frame order matters when reusing the original annotation prompts.
        order = cache.frame_order("article", article.article_id, inventory)
        messages = annotation(
            article.event_name, inventory, order, article.body, article.headline,
        )
        scores = cache.get(request(
            "article_annotation", CONFIG["annotation_model"], messages,
            CONFIG["annotation_replicate"], CONFIG["annotation_settings"],
        ))
        for frame_id, score in scores.items():
            rows.append(dict(
                article_id=article.article_id, frame_id=frame_id,
                frame_name=inventory[frame_id]["name"],
                frame_description=inventory[frame_id]["description"], score=score,
            ))
    save_csv(rows, output_path)
    hits("article_annotation", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    annotate_corpus(args.corpus, args.frames, args.output)

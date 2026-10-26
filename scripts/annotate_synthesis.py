"""Score each successful answer against its event's final frames."""

import argparse

from config import CONFIG
from support.common import Cache, frames, hits, read_csv, request, save_csv
from support.prompts import annotation


def annotate_synthesis(results_path, frames_path, output_path):
    """Score successful answers and write one row per answer-frame pair."""
    cache = Cache()
    inventories = frames(frames_path)
    answers = read_csv(results_path)
    rows = []
    for answer in answers[answers.status.eq("complete")].itertuples():
        inventory = inventories[answer.event_name]
        order = cache.frame_order("answer", answer.synthesis_experiment_id, inventory)
        messages = annotation(answer.event_name, inventory, order, answer.output)
        scores = cache.get(request(
            "annotation", CONFIG["annotation_model"], messages,
            CONFIG["annotation_replicate"], CONFIG["annotation_settings"],
        ))
        for frame_id, score in scores.items():
            rows.append(dict(
                synthesis_experiment_id=answer.synthesis_experiment_id,
                output=answer.output, frame_id=frame_id,
                frame_name=inventory[frame_id]["name"],
                frame_description=inventory[frame_id]["description"], score=score,
            ))
    save_csv(rows, output_path)
    hits(str(output_path), cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    annotate_synthesis(args.results, args.frames, args.output)

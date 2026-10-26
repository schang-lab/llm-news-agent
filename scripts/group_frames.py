"""Group extracted frames before the human editing step, using cached responses when available."""

import argparse

from support.common import Cache, grouping_inputs, hits, save_csv
from support.prompts import grouping_request


def group_frames(extracted_path, output_path):
    """Write proposed groups for each event, adding Other for unmatched frames."""
    cache = Cache("cache/frame_development.jsonl.gz")
    rows = []
    extracted = grouping_inputs(extracted_path)
    for event, articles in extracted.groupby("event_name", sort=False):
        response = cache.get(grouping_request(event, articles.sort_values("position")))
        proposals = response["groups"] + [dict(
            name="Other", description="Frames that fit none of the other groups.",
        )]
        for proposal in proposals:
            rows.append(dict(
                event_name=event, frame_group=proposal["name"],
                frame_group_description=proposal["description"],
            ))
    save_csv(rows, output_path)
    hits("grouping", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-frames", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    group_frames(args.extracted_frames, args.output)

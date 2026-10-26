"""Assign each extracted article frame to one proposed group."""

import argparse

from config import CONFIG
from support.common import Cache, grouping_inputs, hits, read_csv, save_csv
from support.prompts import assignment_request


def assign_frames(extracted_path, groups_path, output_path):
    """Write each extracted frame's group and confidence using batched requests."""
    cache = Cache("cache/frame_development.jsonl.gz")
    rows = []
    extracted = grouping_inputs(extracted_path)
    proposals = read_csv(groups_path)
    batch_size = CONFIG["assignment_batch_size"]
    for event, articles in extracted.groupby("event_name", sort=False):
        articles = articles.sort_values("position")
        event_groups = proposals[
            proposals.event_name.eq(event) & proposals.frame_group.ne("Other")
        ]
        groups = [
            dict(name=row.frame_group, description=row.frame_group_description)
            for row in event_groups.itertuples()
        ]
        # Each prompt contains up to 50 extracted frames, not 50 separate requests.
        for start in range(0, len(articles), batch_size):
            batch = articles.iloc[start:start + batch_size]
            assignments = cache.get(assignment_request(event, batch, groups))["assignments"]
            article_ids = batch.set_index("position").article_id
            for assignment in assignments:
                # The response identifies a frame by its position in the event list.
                rows.append(dict(
                    article_id=article_ids.loc[assignment["frame_id"]], event_name=event,
                    assigned_group=assignment["group"], confidence=assignment["confidence"],
                ))
    save_csv(rows, output_path)
    hits("assignment", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-frames", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    assign_frames(args.extracted_frames, args.groups, args.output)

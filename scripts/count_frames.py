"""Add article counts and shares to frame groups using completed assignments."""

import argparse

from support.common import read_csv, save_csv


def count_frames(groups_path, assignments_path, output_path):
    """Write groups with assignment counts and within-event shares, including Other."""
    groups = read_csv(groups_path)
    assigned = read_csv(assignments_path)
    counts = assigned.groupby(["event_name", "assigned_group"]).size()
    totals = assigned.groupby("event_name").size()
    groups["n_articles"] = [
        int(counts.get((row.event_name, row.frame_group), 0))
        for row in groups.itertuples()
    ]
    # Shares use assigned articles for that event, including the Other group.
    groups["share_of_articles"] = (
        groups.n_articles / groups.event_name.map(totals)
    ).round(4)
    save_csv(groups, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    count_frames(args.groups, args.assignments, args.output)

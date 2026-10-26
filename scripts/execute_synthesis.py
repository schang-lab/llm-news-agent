"""Run synthesis requests, reusing cached model responses when available."""

import argparse

from config import CONFIG
from support.common import Cache, hits, read_csv, read_recorded, request, save_csv
from support.prompts import messages


def execute_synthesis(experiments_path, output_path):
    """Write answers and completion status, using cached responses before provider calls."""
    experiments = read_csv(experiments_path)
    recorded = read_recorded(
        CONFIG["synthesis_result_files"], ["synthesis_experiment_id", "status"],
    )
    failures = set(recorded.loc[recorded.status.ne("complete"), "synthesis_experiment_id"])
    cache = Cache()
    rows = []
    for experiment in experiments.itertuples(index=False):
        status, output = "complete", ""
        # Preserve failed runs rather than silently dropping or retrying them.
        if experiment.synthesis_experiment_id in failures:
            status = "failed"
        else:
            output = cache.get(request(
                "synthesis", experiment.model,
                messages(experiment.system_prompt, experiment.query), experiment.replicate,
                CONFIG["synthesis_settings"][experiment.model],
            ))
        rows.append({**experiment._asdict(), "output": output, "status": status})
    save_csv(rows, output_path)
    hits("synthesis", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", required=True, help="Input experiments: .csv or .csv.gz")
    parser.add_argument("-o", "--output", required=True, help="Output results: .csv or .csv.gz")
    args = parser.parse_args()
    execute_synthesis(args.experiments, args.output)

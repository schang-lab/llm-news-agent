"""Copy study CSVs with article bodies and synthesized answers redacted."""

import argparse
import csv
import gzip
from pathlib import Path
import shutil
import sys

from support.prompts import SYNTHESIS

REDACTED = "[REDACTED]"


def redact_prompt(prompt):
    """Keep the synthesis instructions and redact the full article block."""
    prefix = SYNTHESIS.split("{articles}")[0]
    assert prompt.startswith(prefix)
    return prefix + REDACTED


def create_public_data(data_path, output_path):
    """Write _public CSV copies with bodies, synthesis article blocks and answers redacted."""
    data_path, output_path = Path(data_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # A prompt containing ten articles can exceed the CSV reader's default limit.
    csv.field_size_limit(sys.maxsize)
    for source in sorted(data_path.iterdir()):
        if not source.is_file() or not source.name.endswith((".csv", ".csv.gz")):
            continue
        suffix = ".csv.gz" if source.name.endswith(".csv.gz") else ".csv"
        name = source.name[:-len(suffix)]
        target = output_path / f"{name}_public{suffix}"
        if name not in {
            "corpus", "annotated_synthesis_results", "synthesis_experiments", "synthesis_results",
        }:
            shutil.copy2(source, target)
        else:
            opener = gzip.open if suffix == ".csv.gz" else open
            with opener(source, "rt", encoding="utf-8", newline="") as original:
                reader = csv.DictReader(original)
                with opener(target, "wt", encoding="utf-8", newline="") as public:
                    writer = csv.DictWriter(public, fieldnames=reader.fieldnames, lineterminator="\n")
                    writer.writeheader()
                    for row in reader:
                        if name == "corpus":
                            row["body"] = REDACTED
                        if "output" in row and row["output"]:
                            row["output"] = REDACTED
                        if "system_prompt" in row:
                            row["system_prompt"] = redact_prompt(row["system_prompt"])
                        writer.writerow(row)
        print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    create_public_data(args.data, args.output)

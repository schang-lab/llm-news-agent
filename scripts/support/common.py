"""Shared CSV, cache and experiment helpers."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

import pandas as pd

from . import providers


def read_csv(path, **kwargs):
    """Read plain or compressed CSVs without rounding saved float values."""
    return pd.read_csv(
        path, keep_default_na=False, float_precision="round_trip",
        compression="infer", **kwargs,
    )


def save_csv(rows, path):
    """Write rows to CSV or gzip CSV, creating parent folders, and return the table."""
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="infer")
    print(f"{path}: {len(table):,} rows", flush=True)
    return table


def digest(value):
    """Hash a JSON-serializable value with stable key ordering."""
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def csv_hash(path):
    """Hash CSV contents so compression does not change experiment IDs."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def request(stage, model, messages, replicate=0, settings=None):
    """Describe a request in the same format used to create the cache key."""
    return dict(
        stage=stage, model=model, messages=messages,
        replicate=int(replicate), settings=settings or {},
    )


def corpus(path):
    """Load articles and add numeric source-slant codes for analysis."""
    table = read_csv(path)
    codes = {"Left": -1, "Center": 0, "Right": 1}
    # Resolve mixed source labels using that source's most common single label.
    single = table[table.slant.isin(codes)]
    modes = single.groupby("source").slant.agg(
        lambda labels: Counter(labels).most_common(1)[0][0]
    )
    table["slant_code"] = [
        codes[label if label in codes else modes[source]]
        for label, source in zip(table.slant, table.source)
    ]
    return table


def frames(path):
    """Look up each event's final frame names and descriptions by frame ID."""
    inventories = {}
    for event, group in read_csv(path).groupby("event_name", sort=False):
        inventories[event] = {
            frame.frame_id: dict(name=frame.frame_name, description=frame.frame_description)
            for frame in group.itertuples()
        }
    return inventories


def grouping_inputs(extracted_path):
    """Load extractions and number their positions within each event in CSV row order."""
    # Row order determines both grouping prompts and assignment batches.
    table = read_csv(extracted_path)
    table["position"] = table.groupby("event_name", sort=False).cumcount()
    return table


class Cache:
    """Reuse saved responses; request and save missing responses."""

    def __init__(self, path="cache/responses.jsonl.gz"):
        """Load an optional response archive and individually saved cache entries."""
        path = Path(path)
        self.directory = path.parent / path.name.removesuffix(".jsonl.gz")
        self.records = {}
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                self.records = {record["key"]: record for record in map(json.loads, stream)}
        # New entries are separate files; the original study archive stays unchanged.
        saved = [json.loads(file.read_text()) for file in self.directory.glob("*.json")]
        for record in sorted(saved, key=lambda record: record["order"]):
            self.records[record["key"]] = record
        # Study records retain frame order so annotation prompts can be replayed exactly.
        self.annotation_orders = {}
        for record in self.records.values():
            for item in record.get("annotations", []):
                self.annotation_orders[item["kind"], item["item_id"]] = item["frame_order"]
        self.hits = self.calls = 0

    def record(self, req):
        """Return a full record, requesting and saving it on a miss.

        Cached incomplete or invalid JSON responses raise without another provider call.
        """
        key = digest(req)
        if key in self.records:
            self.hits += 1
        else:
            response = providers.call_model(req)
            self.calls += 1
            record = dict(key=key, request=req, order=len(self.records), **response)
            if not record.get("error"):
                try:
                    raw = record["raw_output"]
                    record["output"] = raw if req["stage"] == "synthesis" else json.loads(raw)
                except json.JSONDecodeError:
                    record["error"] = "The provider response is not valid JSON."
            # Save incomplete/invalid output too, so rerunning does not pay for it again.
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / f"{key}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
            self.records[key] = record
        record = self.records[key]
        if record.get("error"):
            raise ValueError(f"Cached response {key}: {record['error']}")
        return record

    def get(self, req):
        """Return parsed JSON output, or plain text for synthesis."""
        return self.record(req)["output"]

    def frame_order(self, kind, item_id, inventory):
        """Reuse saved frame order, otherwise shuffle reproducibly for this item."""
        order = self.annotation_orders.get((kind, item_id))
        if order is None or set(order) != set(inventory):
            # New prompts get a reproducible shuffle; study prompts keep their saved order.
            order = list(inventory)
            random.Random(f"{kind}:{item_id}").shuffle(order)
        return order


def read_recorded(paths, columns):
    """Use supplied study records when present; new runs need no reference CSVs."""
    tables = [read_csv(path, usecols=columns) for path in paths if Path(path).is_file()]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)


def preserve_order(table, id_column, reference):
    """Keep saved experiments in their original order; append new ones."""
    positions = {key: index for index, key in enumerate(reference[id_column])}
    return (
        table.assign(_position=table[id_column].map(positions))
        .sort_values("_position", kind="stable")
        .drop(columns="_position")
    )


def hits(stage, cache, embeddings=None):
    """Print response/embedding cache hits and new-call counts for one stage."""
    response_hits, model_calls = (cache.hits, cache.calls) if cache else (0, 0)
    embedding_hits, embedding_calls = (embeddings.hits, embeddings.calls) if embeddings else (0, 0)
    print(
        f"{stage}: {response_hits} response-cache hits; {embedding_hits} embedding hits; "
        f"{model_calls} LLM calls; {embedding_calls} embedding calls."
    )

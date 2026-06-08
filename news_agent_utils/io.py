"""Shared IO, serialization, and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_serializable(item) for item in value)
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump(mode="json"))
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    csv_path = Path(data_dir) / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    return pd.read_csv(csv_path, index_col=0)


def save_table(df: pd.DataFrame, data_dir: Path, name: str) -> None:
    csv_path = Path(data_dir) / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    for column in df_to_save.columns:
        df_to_save[column] = df_to_save[column].apply(to_serializable)
    df_to_save.to_csv(csv_path)
    print(f"Saved {name} to {csv_path}")

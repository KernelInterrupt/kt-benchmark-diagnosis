import csv
import os
from functools import lru_cache

import pandas as pd


_MODIN_IMPORT_ERROR = None


def _should_use_modin():
    return os.getenv("PYKT_USE_MODIN", "").strip().lower() in {"1", "true", "yes", "on"}


def _modin_pandas():
    global _MODIN_IMPORT_ERROR
    try:
        import modin.pandas as mpd

        return mpd
    except Exception as exc:
        _MODIN_IMPORT_ERROR = exc
        return None


def read_csv_auto(*args, prefer_modin=False, **kwargs):
    if prefer_modin and _should_use_modin():
        mpd = _modin_pandas()
        if mpd is not None:
            return mpd.read_csv(*args, **kwargs)
        if _MODIN_IMPORT_ERROR is not None:
            print(f"PYKT_USE_MODIN is enabled but Modin is unavailable, fallback to pandas: {_MODIN_IMPORT_ERROR}")
    return pd.read_csv(*args, **kwargs)


@lru_cache(maxsize=256)
def read_csv_columns(sequence_path):
    with open(sequence_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin)
        return tuple(next(reader))


def row_has_value(row, key):
    value = row.get(key)
    return value not in (None, "")


def parse_int_sequence(raw):
    return [int(item) for item in raw.split(",")]


def parse_float_sequence(raw):
    return [float(item) for item in raw.split(",")]


def iter_sequence_rows(sequence_path, folds):
    fold_set = {int(fold) for fold in folds}
    with open(sequence_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            raw_fold = row.get("fold")
            if raw_fold is None:
                continue
            try:
                row_fold = int(raw_fold)
            except (TypeError, ValueError):
                continue
            if row_fold in fold_set:
                yield row

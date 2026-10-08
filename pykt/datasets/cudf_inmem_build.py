from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch


CTW_FLOAT_SEQ_COLS = {
    "ctw_pseqs",
    "ctw_logitseqs",
    "ctw_depthseqs",
    "ctw_totalseqs",
    "ctw_posseqs",
    "ctw_negseqs",
}


def _load_gpu_libs():
    import cupy as cp
    import cudf

    return cp, cudf


def _infer_columns(csv_path: str | Path):
    return set(pd.read_csv(csv_path, nrows=0).columns)


def _infer_uniform_width(list_col):
    if len(list_col) == 0:
        return 0
    lengths = list_col.list.len()
    width_min = int(lengths.min())
    width_max = int(lengths.max())
    if width_min != width_max:
        raise ValueError(f"Non-uniform sequence width detected: min={width_min}, max={width_max}")
    return width_max


def _split_numeric_matrix(series, dtype_name):
    cp, _ = _load_gpu_libs()
    list_col = series.astype("str").str.split(",")
    rows = len(list_col)
    width = _infer_uniform_width(list_col)
    if rows == 0 or width == 0:
        target_dtype = cp.float32 if dtype_name == "float32" else cp.int64
        return cp.empty((rows, width), dtype=target_dtype)
    leaves = list_col.list.leaves.astype(dtype_name).values
    return leaves.reshape((rows, width))


def _tensor_device():
    requested = os.getenv("PYKT_QLEVEL_PREBUILD_TENSOR_DEVICE", "cuda").strip().lower()
    if requested in {"cpu", "host"} or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _cupy_to_torch(cp_array, kind: str, device: torch.device):
    cp, _ = _load_gpu_libs()
    if device.type == "cuda":
        try:
            tensor = torch.from_dlpack(cp_array)
        except Exception:
            tensor = torch.utils.dlpack.from_dlpack(cp_array)
    else:
        tensor = torch.from_numpy(cp.asnumpy(cp_array))
    if kind == "long":
        return tensor.long()
    if kind == "float":
        return tensor.float()
    if kind == "bool":
        return tensor.bool()
    raise ValueError(f"Unsupported tensor kind: {kind}")


def _empty_tensor(kind: str, device: torch.device):
    if kind == "long":
        return torch.empty((0,), dtype=torch.long, device=device)
    if kind == "float":
        return torch.empty((0,), dtype=torch.float32, device=device)
    if kind == "bool":
        return torch.empty((0,), dtype=torch.bool, device=device)
    raise ValueError(f"Unsupported tensor kind: {kind}")


def _build_base_payload(frame, input_type, columns, tensor_device: torch.device, with_qtest: bool):
    cp, _ = _load_gpu_libs()
    rows = len(frame)
    frame_columns = set(frame.columns)
    dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
    if rows == 0:
        dori["qseqs"] = _empty_tensor("long", tensor_device)
        dori["cseqs"] = _empty_tensor("long", tensor_device)
        dori["rseqs"] = _empty_tensor("float", tensor_device)
        dori["tseqs"] = _empty_tensor("long", tensor_device)
        dori["utseqs"] = _empty_tensor("long", tensor_device)
        dori["smasks"] = _empty_tensor("bool", tensor_device)
        dori["masks"] = _empty_tensor("bool", tensor_device)
        dqtest = None
        if with_qtest:
            dqtest = {
                "qidxs": _empty_tensor("long", tensor_device),
                "rests": _empty_tensor("long", tensor_device),
                "orirow": _empty_tensor("long", tensor_device),
            }
        return dori, dqtest, 0

    cp_cache = {}
    if "questions" in frame_columns and ("questions" in input_type or "qidxs" in columns):
        cp_cache["qseqs"] = _split_numeric_matrix(frame["questions"], "int64")
        dori["qseqs"] = _cupy_to_torch(cp_cache["qseqs"], "long", tensor_device)
    else:
        dori["qseqs"] = _empty_tensor("long", tensor_device)

    if "concepts" in input_type and "concepts" in frame_columns:
        cp_cache["cseqs"] = _split_numeric_matrix(frame["concepts"], "int64")
        dori["cseqs"] = _cupy_to_torch(cp_cache["cseqs"], "long", tensor_device)
    else:
        dori["cseqs"] = _empty_tensor("long", tensor_device)

    cp_cache["rseqs"] = _split_numeric_matrix(frame["responses"], "float32")
    cp_cache["smasks_raw"] = _split_numeric_matrix(frame["selectmasks"], "int64")
    dori["rseqs"] = _cupy_to_torch(cp_cache["rseqs"], "float", tensor_device)

    if "timestamps" in frame_columns:
        cp_cache["tseqs"] = _split_numeric_matrix(frame["timestamps"], "int64")
        dori["tseqs"] = _cupy_to_torch(cp_cache["tseqs"], "long", tensor_device)
    else:
        dori["tseqs"] = _empty_tensor("long", tensor_device)

    if "usetimes" in frame_columns:
        cp_cache["utseqs"] = _split_numeric_matrix(frame["usetimes"], "int64")
        dori["utseqs"] = _cupy_to_torch(cp_cache["utseqs"], "long", tensor_device)
    else:
        dori["utseqs"] = _empty_tensor("long", tensor_device)

    if "uid" in frame_columns:
        dori["uid"] = _cupy_to_torch(frame["uid"].astype("int64").values, "long", tensor_device)

    for col in sorted(CTW_FLOAT_SEQ_COLS.intersection(frame_columns)):
        cp_cache[col] = _split_numeric_matrix(frame[col], "float32")
        dori[col] = _cupy_to_torch(cp_cache[col], "float", tensor_device)

    interaction_num = int(cp.count_nonzero(cp_cache["smasks_raw"] == 1).item())
    mask_source = cp_cache.get("cseqs", cp_cache.get("qseqs"))
    if mask_source is None:
        raise ValueError("in-memory cudf build requires questions or concepts as mask source")
    dori["masks"] = _cupy_to_torch((mask_source[:, :-1] != -1) & (mask_source[:, 1:] != -1), "bool", tensor_device)
    dori["smasks"] = _cupy_to_torch(cp_cache["smasks_raw"][:, 1:] != -1, "bool", tensor_device)

    dqtest = None
    if with_qtest:
        dqtest = {
            "qidxs": _cupy_to_torch(_split_numeric_matrix(frame["qidxs"], "int64")[:, 1:], "long", tensor_device),
            "rests": _cupy_to_torch(_split_numeric_matrix(frame["rest"], "int64")[:, 1:], "long", tensor_device),
            "orirow": _cupy_to_torch(_split_numeric_matrix(frame["orirow"], "int64")[:, 1:], "long", tensor_device),
        }
    return dori, dqtest, interaction_num


def build_sequence_cache_inmem(csv_path: str | Path, input_type, folds, device: int = 0):
    cp, cudf = _load_gpu_libs()
    columns = _infer_columns(csv_path)
    required_cols = ["fold", "responses", "selectmasks"]
    if "questions" in columns and "questions" in input_type:
        required_cols.append("questions")
    if "concepts" in columns and "concepts" in input_type:
        required_cols.append("concepts")
    if "timestamps" in columns:
        required_cols.append("timestamps")
    if "usetimes" in columns:
        required_cols.append("usetimes")
    if "uid" in columns:
        required_cols.append("uid")
    required_cols.extend(sorted(CTW_FLOAT_SEQ_COLS.intersection(columns)))
    tensor_device = _tensor_device()
    with cp.cuda.Device(int(device)):
        frame = cudf.read_csv(csv_path, usecols=required_cols)
        frame = frame[frame["fold"].isin(list(folds))]
        dori, _, interaction_num = _build_base_payload(frame, input_type, columns, tensor_device, with_qtest=False)
    return dori, interaction_num


def build_qtest_cache_inmem(csv_path: str | Path, input_type, folds, device: int = 0):
    cp, cudf = _load_gpu_libs()
    columns = _infer_columns(csv_path)
    required_cols = ["fold", "responses", "selectmasks", "qidxs", "rest", "orirow"]
    if "questions" in columns:
        required_cols.append("questions")
    if "concepts" in columns:
        required_cols.append("concepts")
    if "timestamps" in columns:
        required_cols.append("timestamps")
    if "usetimes" in columns:
        required_cols.append("usetimes")
    if "uid" in columns:
        required_cols.append("uid")
    required_cols.extend(sorted(CTW_FLOAT_SEQ_COLS.intersection(columns)))
    tensor_device = _tensor_device()
    with cp.cuda.Device(int(device)):
        frame = cudf.read_csv(csv_path, usecols=required_cols)
        frame = frame[frame["fold"].isin(list(folds))]
        dori, dqtest, interaction_num = _build_base_payload(frame, input_type, columns, tensor_device, with_qtest=True)
    return dori, dqtest, interaction_num

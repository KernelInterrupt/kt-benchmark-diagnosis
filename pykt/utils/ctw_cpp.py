from __future__ import annotations

import copy
import ctypes
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from .ctw_estimator import CTWPredictionStats


_THIS_DIR = Path(__file__).resolve().parent
_SRC_PATH = _THIS_DIR / "ctw_core.cpp"
_LIB_PATH = _THIS_DIR / "_ctw_core.so"


def build_ctw_cpp_shared(force: bool = False) -> Path:
    if _LIB_PATH.exists() and not force and _LIB_PATH.stat().st_mtime >= _SRC_PATH.stat().st_mtime:
        return _LIB_PATH
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(_SRC_PATH),
        "-o",
        str(_LIB_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(_THIS_DIR))
    return _LIB_PATH


def _load_ctw_lib() -> ctypes.CDLL:
    lib_path = build_ctw_cpp_shared(force=False)
    lib = ctypes.CDLL(str(lib_path))

    lib.ctw_last_error.restype = ctypes.c_char_p

    lib.ctw_create.argtypes = [ctypes.c_int]
    lib.ctw_create.restype = ctypes.c_void_p

    lib.ctw_destroy.argtypes = [ctypes.c_void_p]
    lib.ctw_destroy.restype = None

    lib.ctw_clone.argtypes = [ctypes.c_void_p]
    lib.ctw_clone.restype = ctypes.c_void_p

    lib.ctw_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ctw_predict.restype = ctypes.c_int

    lib.ctw_update.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ctw_update.restype = ctypes.c_int

    lib.ctw_checkpoint.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.ctw_checkpoint.restype = ctypes.c_int

    lib.ctw_rollback.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ctw_rollback.restype = ctypes.c_int

    return lib


class CppCTWBinaryKTContextEstimator:
    def __init__(
        self,
        max_depth: int = 6,
        *,
        _lib: Optional[ctypes.CDLL] = None,
        _handle: Optional[int] = None,
        _token_to_id: Optional[Dict[str, int]] = None,
        _next_id: int = 1,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        self.max_depth = int(max_depth)
        self._lib = _lib or _load_ctw_lib()
        self._token_to_id: Dict[str, int] = dict(_token_to_id or {})
        self._next_id = int(_next_id)
        self._handle = int(_handle) if _handle is not None else 0
        if self._handle == 0:
            self._handle = int(self._lib.ctw_create(self.max_depth))
            if self._handle == 0:
                raise RuntimeError(self._last_error())

    def _last_error(self) -> str:
        raw = self._lib.ctw_last_error()
        if raw is None:
            return "Unknown CTW C++ error"
        return raw.decode("utf8", errors="replace")

    def _token_id(self, token: str) -> int:
        token = str(token)
        idx = self._token_to_id.get(token)
        if idx is None:
            idx = self._next_id
            self._token_to_id[token] = idx
            self._next_id += 1
        return idx

    def _suffix_rev_array(self, context_tokens: Sequence[str]):
        if len(context_tokens) <= self.max_depth:
            suffix = list(reversed([self._token_id(tok) for tok in context_tokens]))
        else:
            suffix = list(reversed([self._token_id(tok) for tok in context_tokens[-self.max_depth :]]))
        if len(suffix) == 0:
            return None, 0
        arr = (ctypes.c_int * len(suffix))(*suffix)
        return arr, len(suffix)

    def clone(self) -> "CppCTWBinaryKTContextEstimator":
        cloned_handle = int(self._lib.ctw_clone(self._handle))
        if cloned_handle == 0:
            raise RuntimeError(self._last_error())
        return CppCTWBinaryKTContextEstimator(
            max_depth=self.max_depth,
            _lib=self._lib,
            _handle=cloned_handle,
            _token_to_id=copy.deepcopy(self._token_to_id),
            _next_id=self._next_id,
        )

    def predict(self, context_tokens: Sequence[str]) -> CTWPredictionStats:
        arr, length = self._suffix_rev_array(context_tokens)
        p_correct = ctypes.c_double()
        deepest_depth = ctypes.c_int()
        deepest_total = ctypes.c_int()
        deepest_positive = ctypes.c_int()
        deepest_negative = ctypes.c_int()
        status = self._lib.ctw_predict(
            self._handle,
            arr,
            length,
            ctypes.byref(p_correct),
            ctypes.byref(deepest_depth),
            ctypes.byref(deepest_total),
            ctypes.byref(deepest_positive),
            ctypes.byref(deepest_negative),
        )
        if status != 0:
            raise RuntimeError(self._last_error())
        return CTWPredictionStats(
            p_correct=float(p_correct.value),
            deepest_match_depth=int(deepest_depth.value),
            deepest_match_total=int(deepest_total.value),
            deepest_match_positive=int(deepest_positive.value),
            deepest_match_negative=int(deepest_negative.value),
        )

    def update(self, context_tokens: Sequence[str], target: int) -> None:
        if target not in (0, 1):
            raise ValueError("target must be binary.")
        arr, length = self._suffix_rev_array(context_tokens)
        status = self._lib.ctw_update(self._handle, arr, length, int(target))
        if status != 0:
            raise RuntimeError(self._last_error())

    def checkpoint(self) -> int:
        marker = ctypes.c_size_t()
        status = self._lib.ctw_checkpoint(self._handle, ctypes.byref(marker))
        if status != 0:
            raise RuntimeError(self._last_error())
        return int(marker.value)

    def rollback(self, marker: int) -> None:
        status = self._lib.ctw_rollback(self._handle, int(marker))
        if status != 0:
            raise RuntimeError(self._last_error())

    def close(self) -> None:
        if getattr(self, "_handle", 0):
            self._lib.ctw_destroy(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

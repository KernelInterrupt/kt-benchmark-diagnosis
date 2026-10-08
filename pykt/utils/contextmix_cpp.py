from __future__ import annotations

import copy
import ctypes
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from .contextmix_estimator import ContextMixPredictionStats


_THIS_DIR = Path(__file__).resolve().parent
_SRC_PATH = _THIS_DIR / "contextmix_core.cpp"
_LIB_PATH = _THIS_DIR / "_contextmix_core.so"


def build_contextmix_cpp_shared(force: bool = False) -> Path:
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


def _load_contextmix_lib() -> ctypes.CDLL:
    lib_path = build_contextmix_cpp_shared(force=False)
    lib = ctypes.CDLL(str(lib_path))

    lib.contextmix_last_error.restype = ctypes.c_char_p

    lib.contextmix_create.argtypes = [ctypes.c_int, ctypes.c_double]
    lib.contextmix_create.restype = ctypes.c_void_p

    lib.contextmix_destroy.argtypes = [ctypes.c_void_p]
    lib.contextmix_destroy.restype = None

    lib.contextmix_clone.argtypes = [ctypes.c_void_p]
    lib.contextmix_clone.restype = ctypes.c_void_p

    lib.contextmix_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.contextmix_predict.restype = ctypes.c_int

    lib.contextmix_update.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.contextmix_update.restype = ctypes.c_int

    lib.contextmix_checkpoint.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.contextmix_checkpoint.restype = ctypes.c_int

    lib.contextmix_rollback.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.contextmix_rollback.restype = ctypes.c_int

    return lib


class CppContextMixBinaryKTContextEstimator:
    def __init__(
        self,
        max_order: int = 6,
        learning_rate: float = 0.5,
        *,
        _lib: Optional[ctypes.CDLL] = None,
        _handle: Optional[int] = None,
        _token_to_id: Optional[Dict[str, int]] = None,
        _next_id: int = 1,
    ) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.max_order = int(max_order)
        self.learning_rate = float(learning_rate)
        self._lib = _lib or _load_contextmix_lib()
        self._token_to_id: Dict[str, int] = dict(_token_to_id or {})
        self._next_id = int(_next_id)
        self._handle = int(_handle) if _handle is not None else 0
        if self._handle == 0:
            self._handle = int(self._lib.contextmix_create(self.max_order, self.learning_rate))
            if self._handle == 0:
                raise RuntimeError(self._last_error())

    def _last_error(self) -> str:
        raw = self._lib.contextmix_last_error()
        if raw is None:
            return "Unknown ContextMix C++ error"
        return raw.decode("utf8", errors="replace")

    def _token_id(self, token: str) -> int:
        token = str(token)
        idx = self._token_to_id.get(token)
        if idx is None:
            idx = self._next_id
            self._token_to_id[token] = idx
            self._next_id += 1
        return idx

    def _suffix_array(self, context_tokens: Sequence[str]):
        if self.max_order == 0:
            return None, 0
        tokens = [self._token_id(tok) for tok in context_tokens[-self.max_order :]]
        if len(tokens) == 0:
            return None, 0
        arr = (ctypes.c_int * len(tokens))(*tokens)
        return arr, len(tokens)

    def clone(self) -> "CppContextMixBinaryKTContextEstimator":
        cloned_handle = int(self._lib.contextmix_clone(self._handle))
        if cloned_handle == 0:
            raise RuntimeError(self._last_error())
        return CppContextMixBinaryKTContextEstimator(
            max_order=self.max_order,
            learning_rate=self.learning_rate,
            _lib=self._lib,
            _handle=cloned_handle,
            _token_to_id=copy.deepcopy(self._token_to_id),
            _next_id=self._next_id,
        )

    def predict(self, context_tokens: Sequence[str]) -> ContextMixPredictionStats:
        arr, length = self._suffix_array(context_tokens)
        p_correct = ctypes.c_double()
        deepest_depth = ctypes.c_int()
        deepest_total = ctypes.c_int()
        deepest_positive = ctypes.c_int()
        deepest_negative = ctypes.c_int()
        status = self._lib.contextmix_predict(
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
        return ContextMixPredictionStats(
            p_correct=float(p_correct.value),
            deepest_match_depth=int(deepest_depth.value),
            deepest_match_total=int(deepest_total.value),
            deepest_match_positive=int(deepest_positive.value),
            deepest_match_negative=int(deepest_negative.value),
        )

    def update(self, context_tokens: Sequence[str], target: int) -> None:
        if target not in (0, 1):
            raise ValueError("target must be binary.")
        arr, length = self._suffix_array(context_tokens)
        status = self._lib.contextmix_update(self._handle, arr, length, int(target))
        if status != 0:
            raise RuntimeError(self._last_error())

    def checkpoint(self) -> int:
        marker = ctypes.c_size_t()
        status = self._lib.contextmix_checkpoint(self._handle, ctypes.byref(marker))
        if status != 0:
            raise RuntimeError(self._last_error())
        return int(marker.value)

    def rollback(self, marker: int) -> None:
        status = self._lib.contextmix_rollback(self._handle, int(marker))
        if status != 0:
            raise RuntimeError(self._last_error())

    def close(self) -> None:
        if getattr(self, "_handle", 0):
            self._lib.contextmix_destroy(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

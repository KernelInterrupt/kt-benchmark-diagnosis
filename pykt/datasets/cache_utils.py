import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

import fcntl
import pandas as pd


@contextmanager
def cache_lock(cache_path):
    lock_path = Path(str(cache_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def atomic_pickle_dump(payload, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=cache_path.parent)
    os.close(fd)
    try:
        pd.to_pickle(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _qlevel_prebuild_mode() -> str:
    return os.getenv("PYKT_QLEVEL_PREBUILD_MODE", "auto").strip().lower()


def _force_gpu_prebuild() -> bool:
    return _truthy_env("PYKT_CUDF_PREBUILD_FORCE", default=False)


def _inmemory_gpu_prebuild() -> bool:
    return _truthy_env("PYKT_QLEVEL_PREBUILD_INMEM", default=False)


def _effective_prebuild_device() -> str:
    raw = os.getenv("PYKT_QLEVEL_PREBUILD_DEVICE", "0").strip()
    try:
        requested = int(raw)
    except ValueError:
        requested = 0

    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = [item.strip() for item in visible.split(",") if item.strip()]
        if tokens:
            if 0 <= requested < len(tokens):
                return str(requested)
            return "0"
    return str(max(0, requested))


def _guess_qlevel_prebuild_python() -> str | None:
    override = os.getenv("PYKT_QLEVEL_PREBUILD_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return str(candidate)

    if find_spec("cudf") is not None:
        return sys.executable

    conda_exe = os.getenv("CONDA_EXE")
    env_name = os.getenv("PYKT_QLEVEL_PREBUILD_ENV", "pykt-core-env")
    if conda_exe:
        root = Path(conda_exe).resolve().parent.parent
        candidate = root / "envs" / env_name / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return None


def try_gpu_qlevel_prebuild(csv_path, folds, input_type, max_concepts: int, processed_data) -> bool:
    mode = _qlevel_prebuild_mode()
    if mode in {"none", "off", "false", "0", "cpu", "disabled"}:
        return False

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "tools" / "precompute_qlevel_cache_cudf.py"
    python_bin = _guess_qlevel_prebuild_python()
    if not script_path.exists() or not python_bin:
        if mode == "gpu":
            print(f"GPU qlevel prebuild unavailable: script={script_path.exists()} python={bool(python_bin)}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYKT_QLEVEL_PREBUILD_DEVICE"] = _effective_prebuild_device()

    cmd = [
        python_bin,
        str(script_path),
        "--csv",
        str(csv_path),
        "--folds",
        ",".join(str(item) for item in folds),
        "--input-type",
        ",".join(input_type),
        "--max-concepts",
        str(int(max_concepts)),
        "--device",
        env["PYKT_QLEVEL_PREBUILD_DEVICE"],
    ]
    if _force_gpu_prebuild():
        cmd.append("--force")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    except Exception as exc:
        print(f"GPU qlevel prebuild launch failed for {csv_path}: {exc}")
        return False

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"returncode={proc.returncode}"
        print(f"GPU qlevel prebuild failed for {csv_path}: {detail}")
        return False

    if _truthy_env("PYKT_QLEVEL_PREBUILD_VERBOSE", default=False):
        output = (proc.stdout or "").strip()
        if output:
            print(output)
    return Path(processed_data).exists()


def try_gpu_qtest_prebuild(csv_path, folds, input_type, processed_data, variant: str = "base", extra_args=None) -> bool:
    mode = _qlevel_prebuild_mode()
    if mode in {"none", "off", "false", "0", "cpu", "disabled"}:
        return False

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "tools" / "precompute_qtest_cache_cudf.py"
    python_bin = _guess_qlevel_prebuild_python()
    if not script_path.exists() or not python_bin:
        if mode == "gpu":
            print(f"GPU qtest prebuild unavailable: script={script_path.exists()} python={bool(python_bin)}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYKT_QLEVEL_PREBUILD_DEVICE"] = _effective_prebuild_device()

    cmd = [
        python_bin,
        str(script_path),
        "--csv",
        str(csv_path),
        "--folds",
        ",".join(str(item) for item in folds),
        "--input-type",
        ",".join(input_type),
        "--variant",
        variant,
        "--device",
        env["PYKT_QLEVEL_PREBUILD_DEVICE"],
    ]
    if _force_gpu_prebuild():
        cmd.append("--force")
    if extra_args:
        for key, value in extra_args.items():
            cmd.extend([str(key), str(value)])
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    except Exception as exc:
        print(f"GPU qtest prebuild launch failed for {csv_path}: {exc}")
        return False

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"returncode={proc.returncode}"
        print(f"GPU qtest prebuild failed for {csv_path}: {detail}")
        return False

    if _truthy_env("PYKT_QLEVEL_PREBUILD_VERBOSE", default=False):
        output = (proc.stdout or "").strip()
        if output:
            print(output)
    return Path(processed_data).exists()


def try_gpu_sequence_prebuild(csv_path, folds, input_type, processed_data) -> bool:
    mode = _qlevel_prebuild_mode()
    if mode in {"none", "off", "false", "0", "cpu", "disabled"}:
        return False

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "tools" / "precompute_sequence_cache_cudf.py"
    python_bin = _guess_qlevel_prebuild_python()
    if not script_path.exists() or not python_bin:
        if mode == "gpu":
            print(f"GPU sequence prebuild unavailable: script={script_path.exists()} python={bool(python_bin)}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYKT_QLEVEL_PREBUILD_DEVICE"] = _effective_prebuild_device()

    cmd = [
        python_bin,
        str(script_path),
        "--csv",
        str(csv_path),
        "--folds",
        ",".join(str(item) for item in folds),
        "--input-type",
        ",".join(input_type),
        "--device",
        env["PYKT_QLEVEL_PREBUILD_DEVICE"],
    ]
    if _force_gpu_prebuild():
        cmd.append("--force")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    except Exception as exc:
        print(f"GPU sequence prebuild launch failed for {csv_path}: {exc}")
        return False

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"returncode={proc.returncode}"
        print(f"GPU sequence prebuild failed for {csv_path}: {detail}")
        return False

    if _truthy_env("PYKT_QLEVEL_PREBUILD_VERBOSE", default=False):
        output = (proc.stdout or "").strip()
        if output:
            print(output)
    return Path(processed_data).exists()

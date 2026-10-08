#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import itertools
import json
import math
import os
import random
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

train_main = None
ALIAS_FILES = None
DATASET_PATHS = None
build_prebuild_cmd = None


ALL_FOLDS = (0, 1, 2, 3, 4)
PROJECT_NAME = "pykt-simplekt-residual-ctw-local-bayes"
SEARCH_SPACE = {
    "seed": [42, 3407],
    "dropout": [0.05, 0.1, 0.2],
    "final_fc_dim": [128, 256],
    "final_fc_dim2": [64, 128, 256],
    "d_model": [64, 128],
    "n_blocks": [2, 4],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [64, 128, 256],
    "ctw_feat_dim": [64, 128],
    "ctw_delta_scale": [0.5, 1.0, 2.0],
}
FIXED_PARAMS = {
    "model_name": "simplekt_residual",
    "emb_type": "qid",
    "num_layers": 2,
    "nheads": 4,
    "loss1": 0.5,
    "loss2": 0.5,
    "loss3": 0.5,
    "start": 50,
    "d_ff": 256,
    "num_attn_heads": 4,
    "num_epochs": 50,
    "early_stop_patience": 4,
    "selection_metric": "validnll",
    "use_wandb": 1,
    "add_uuid": 1,
}
GPU_TASKS = {
    0: [
        ("algebra2005", 0),
        ("algebra2005", 1),
        ("nips_task34", 0),
        ("nips_task34", 1),
        ("nips_task34", 2),
    ],
    1: [
        ("algebra2005", 2),
        ("algebra2005", 3),
        ("algebra2005", 4),
        ("nips_task34", 3),
        ("nips_task34", 4),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Local Bayes worker with persistent dataset cache.")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--trials-per-task", type=int, default=8)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--prebuild-python", default=os.environ.get("PYKT_QLEVEL_PREBUILD_PYTHON", sys.executable))
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--only", default="", help="Comma-separated subset like nips_task34:0,algebra2005:1")
    return parser.parse_args()


def relpath(path: Path) -> str:
    return os.path.relpath(path, REPO_ROOT)


def job_base(dataset_name: str, fold: int) -> dict:
    if DATASET_PATHS is None or ALIAS_FILES is None:
        raise RuntimeError("worker imports not initialized")
    dpath = DATASET_PATHS[dataset_name]
    return {
        "dataset_name": dataset_name,
        "fold": int(fold),
        "save_dir": str(REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes" / dataset_name / "simplekt_residual"),
        "train_valid_file_override": relpath(dpath / ALIAS_FILES["train_valid_sequences.csv"]),
        "test_file_override": relpath(dpath / ALIAS_FILES["test_sequences.csv"]),
        "test_window_file_override": relpath(dpath / ALIAS_FILES["test_window_sequences.csv"]),
        "test_question_file_override": relpath(dpath / ALIAS_FILES["test_question_sequences.csv"]),
        "test_question_window_file_override": relpath(dpath / ALIAS_FILES["test_question_window_sequences.csv"]),
        **FIXED_PARAMS,
    }


def required_train_specs(job: dict):
    train_folds = tuple(item for item in ALL_FOLDS if item != int(job["fold"]))
    valid_folds = (int(job["fold"]),)
    return [
        {
            "mode": "sequence",
            "csv": job["train_valid_file_override"],
            "folds": valid_folds,
            "input_type": "questions,concepts",
            "output": str(Path(str(REPO_ROOT / job["train_valid_file_override"]) + "_" + str(job["fold"]) + ".pkl")),
        },
        {
            "mode": "sequence",
            "csv": job["train_valid_file_override"],
            "folds": train_folds,
            "input_type": "questions,concepts",
            "output": str(Path(str(REPO_ROOT / job["train_valid_file_override"]) + "_" + "_".join(str(x) for x in train_folds) + ".pkl")),
        },
    ]


def prebuild_missing(tasks: list[tuple[str, int]], python_bin: str, gpu_id: int):
    if build_prebuild_cmd is None:
        raise RuntimeError("worker imports not initialized")
    seen = set()
    for dataset_name, fold in tasks:
        job = job_base(dataset_name, fold)
        for spec in required_train_specs(job):
            key = (spec["csv"], tuple(spec["folds"]), spec["mode"])
            if key in seen:
                continue
            seen.add(key)
            output = Path(spec["output"])
            if output.exists():
                continue
            cmd = build_prebuild_cmd(spec, python_bin, device=gpu_id)
            print(f"[prebuild] {' '.join(cmd)}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            rc = os.spawnve(os.P_WAIT, python_bin, cmd, env)
            if rc != 0:
                raise SystemExit(f"prebuild failed for {output} with code={rc}")


class LocalBayesTask:
    def __init__(self, dataset_name: str, fold: int, trials_per_task: int, rng: random.Random):
        self.dataset_name = dataset_name
        self.fold = int(fold)
        self.trials_per_task = int(trials_per_task)
        self.rng = rng
        self.space_keys = list(SEARCH_SPACE.keys())
        self.candidates = list(itertools.product(*(SEARCH_SPACE[key] for key in self.space_keys)))
        self.tried: set[tuple] = set()
        self.records: list[dict] = []

    @property
    def tag(self) -> str:
        return f"{self.dataset_name}_fold{self.fold}"

    def done(self) -> bool:
        return len(self.records) >= self.trials_per_task or len(self.tried) >= len(self.candidates)

    def _encode(self, candidate: tuple) -> list[float]:
        coords = []
        for key, value in zip(self.space_keys, candidate):
            values = SEARCH_SPACE[key]
            if len(values) == 1:
                coords.append(0.0)
            else:
                coords.append(values.index(value) / float(len(values) - 1))
        return coords

    def _sample_random(self) -> tuple:
        remaining = [candidate for candidate in self.candidates if candidate not in self.tried]
        return self.rng.choice(remaining)

    def propose(self) -> tuple:
        if len(self.records) < 3:
            candidate = self._sample_random()
            self.tried.add(candidate)
            return candidate

        xs = np.asarray([self._encode(record["candidate"]) for record in self.records], dtype=np.float64)
        ys = np.asarray([record["objective"] for record in self.records], dtype=np.float64)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(xs.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=0)
        gp.fit(xs, ys)

        candidate_pool = [candidate for candidate in self.candidates if candidate not in self.tried]
        if not candidate_pool:
            raise RuntimeError("No candidate left")
        if len(candidate_pool) > 128:
            candidate_pool = self.rng.sample(candidate_pool, 128)
        cand_x = np.asarray([self._encode(candidate) for candidate in candidate_pool], dtype=np.float64)
        mean, std = gp.predict(cand_x, return_std=True)
        best_y = float(np.min(ys))
        improvement = best_y - mean
        z = improvement / np.maximum(std, 1e-9)
        # Normal CDF/PDF without scipy.
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        ei = improvement * cdf + std * pdf
        best_idx = int(np.argmax(ei))
        candidate = candidate_pool[best_idx]
        self.tried.add(candidate)
        return candidate

    def build_params(self, candidate: tuple, trial_index: int) -> dict:
        params = job_base(self.dataset_name, self.fold)
        params.update(dict(zip(self.space_keys, candidate)))
        params["project_name"] = PROJECT_NAME
        params["wandb_group"] = self.tag
        params["wandb_job_type"] = "local_bayes_worker"
        params["wandb_name"] = f"{self.tag}_trial{trial_index}"
        return params

    def register(self, candidate: tuple, result: dict):
        self.records.append(
            {
                "candidate": candidate,
                "objective": float(result["validnll"]),
                "result": result,
            }
        )


class ThreadSafeDatasetCache:
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def clear(self):
        with self._lock:
            self._data.clear()


def persist_record(log_path: Path, payload: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_trial(slot_id: int, params: dict, dataset_cache, cuda_stream, concurrent_mode: bool):
    import torch

    local_params = dict(params)
    if concurrent_mode:
        local_params["use_wandb"] = 0
    ctx = nullcontext()
    if cuda_stream is not None:
        ctx = torch.cuda.stream(cuda_stream)
    with ctx:
        result = train_main(local_params, dataset_cache=dataset_cache)
        if cuda_stream is not None:
            cuda_stream.synchronize()
    result["slot_id"] = slot_id
    return result


def main():
    global train_main, ALIAS_FILES, DATASET_PATHS, build_prebuild_cmd
    args = parse_args()
    if args.gpu_id not in GPU_TASKS:
        raise SystemExit(f"Unsupported gpu-id: {args.gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from examples.wandb_train import main as train_main_impl
    from tools.run_simplekt_residual_ctw_queue import (
        ALIAS_FILES as alias_files_impl,
        DATASET_PATHS as dataset_paths_impl,
        build_prebuild_cmd as build_prebuild_cmd_impl,
    )

    train_main = train_main_impl
    ALIAS_FILES = alias_files_impl
    DATASET_PATHS = dataset_paths_impl
    build_prebuild_cmd = build_prebuild_cmd_impl
    os.environ.setdefault("PYKT_QLEVEL_PREBUILD_MODE", "gpu")
    os.environ.setdefault("PYKT_QLEVEL_PREBUILD_DEVICE", "0")
    os.environ.setdefault("PYKT_TORCH_COMPILE", "0")
    os.environ.setdefault("PYKT_TORCH_AMP", "1")
    os.environ.setdefault("PYKT_TORCH_AMP_DTYPE", "float16")
    os.environ.setdefault("PYKT_PIN_MEMORY", "0")
    os.environ.setdefault("PYKT_TRAIN_DATASET_DEVICE", "cuda")
    os.environ.setdefault("PYKT_VALID_DATASET_DEVICE", "cuda")
    os.environ.setdefault("PYKT_TEST_DATASET_DEVICE", "cuda")
    os.environ.setdefault("PYKT_TRAIN_NUM_WORKERS", "0")
    os.environ.setdefault("PYKT_VALID_NUM_WORKERS", "0")
    os.environ.setdefault("PYKT_TEST_NUM_WORKERS", "0")
    os.environ["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    rng = random.Random(args.seed + args.gpu_id)
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    task_pairs = []
    for dataset_name, fold in GPU_TASKS[args.gpu_id]:
        token = f"{dataset_name}:{fold}"
        if selected and token not in selected:
            continue
        task_pairs.append((dataset_name, fold))
    max_slots = max(1, int(args.slots))
    tasks = [LocalBayesTask(dataset_name, fold, args.trials_per_task, rng) for dataset_name, fold in task_pairs]
    dataset_cache = ThreadSafeDatasetCache()
    prebuild_missing(task_pairs, args.prebuild_python, args.gpu_id)

    status_path = REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes" / f"gpu{args.gpu_id}_status.jsonl"
    streams = [torch.cuda.Stream(device=0) for _ in range(max_slots)] if torch.cuda.is_available() else [None] * max_slots
    print(
        json.dumps(
            {
                "worker": "local_bayes",
                "gpu_id": args.gpu_id,
                "slots": max_slots,
                "tasks": [task.tag for task in tasks],
            }
        ),
        flush=True,
    )

    running = {}
    available_slots = list(range(max_slots))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_slots, thread_name_prefix=f"gpu{args.gpu_id}-slot") as executor:
        while True:
            running_tags = {meta["task"].tag for meta in running.values()}
            pending = [task for task in tasks if not task.done() and task.tag not in running_tags]
            pending.sort(key=lambda task: (len(task.records), task.dataset_name, task.fold))

            while available_slots and pending:
                slot_id = available_slots.pop(0)
                task = pending.pop(0)
                trial_index = len(task.records) + 1
                candidate = task.propose()
                params = task.build_params(candidate, trial_index)
                print(
                    f"[trial-start] gpu={args.gpu_id} slot={slot_id} task={task.tag} "
                    f"trial={trial_index} candidate={candidate}",
                    flush=True,
                )
                start_time = time.time()
                future = executor.submit(
                    run_trial,
                    slot_id,
                    params,
                    dataset_cache,
                    streams[slot_id] if slot_id < len(streams) else None,
                    max_slots > 1,
                )
                running[future] = {
                    "slot_id": slot_id,
                    "task": task,
                    "trial_index": trial_index,
                    "candidate": candidate,
                    "start_time": start_time,
                }

            if not running:
                break

            done, _ = concurrent.futures.wait(
                tuple(running.keys()),
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue

            for future in done:
                meta = running.pop(future)
                available_slots.append(meta["slot_id"])
                available_slots.sort()
                result = future.result()
                duration = time.time() - meta["start_time"]
                result["duration_sec"] = round(duration, 3)
                result["task"] = meta["task"].tag
                result["trial_index"] = meta["trial_index"]
                result["candidate"] = dict(zip(meta["task"].space_keys, meta["candidate"]))
                meta["task"].register(meta["candidate"], result)
                persist_record(status_path, result)
                print(
                    f"[trial-end] gpu={args.gpu_id} slot={meta['slot_id']} task={meta['task'].tag} "
                    f"trial={meta['trial_index']} validnll={result['validnll']:.6f} "
                    f"validauc={result['validauc']:.6f} duration={duration:.1f}s",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1)


if __name__ == "__main__":
    main()

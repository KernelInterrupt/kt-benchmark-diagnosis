#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("PYKT_RUN_PYTHON", "").strip() or sys.executable
TRAIN_PARAM_ORDER = [
    "dataset_name",
    "model_name",
    "emb_type",
    "save_dir",
    "seed",
    "fold",
    "dropout",
    "final_fc_dim",
    "final_fc_dim2",
    "num_layers",
    "nheads",
    "loss1",
    "loss2",
    "loss3",
    "start",
    "d_model",
    "d_ff",
    "num_attn_heads",
    "n_blocks",
    "learning_rate",
    "batch_size",
    "num_epochs",
    "early_stop_patience",
    "ctw_feat_dim",
    "ctw_delta_scale",
    "train_valid_file_override",
    "test_file_override",
    "test_window_file_override",
    "test_question_file_override",
    "test_question_window_file_override",
    "use_wandb",
    "add_uuid",
]

DATASET_PATHS = {
    "nips_task34": REPO_ROOT / "data" / "nips_task34" / "train_data",
    "algebra2005": REPO_ROOT / "data" / "algebra2005",
    "assist2012": REPO_ROOT / "data" / "assist2012",
}

ALIAS_FILES = {
    "train_valid_sequences.csv": "tv_ctw6.csv",
    "test_sequences.csv": "test_ctw6.csv",
    "test_window_sequences.csv": "testw_ctw6.csv",
    "test_question_sequences.csv": "tq_ctw6.csv",
    "test_question_window_sequences.csv": "tqw_ctw6.csv",
}


def relpath(path: Path) -> str:
    return os.path.relpath(path, REPO_ROOT)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def safe_param_values(params: dict) -> list[str]:
    values = []
    for key in TRAIN_PARAM_ORDER:
        if key not in params:
            continue
        value = params[key]
        values.append(str(value).replace("/", "_"))
    return values


def ckpt_dir(save_dir: Path, params: dict) -> Path:
    params_str = "_".join(safe_param_values(params))
    if len(params_str) > 180:
        params_hash = hashlib.sha1(params_str.encode("utf-8")).hexdigest()[:16]
        params_str = params_str[:120] + "__" + params_hash
    return save_dir / params_str


def run_checked(cmd: list[str], log_path: Path | None = None):
    if log_path is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write(f"\n$ {' '.join(cmd)}\n")
        fout.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=fout, stderr=subprocess.STDOUT)


def ensure_augmented_dataset(dataset: str, depth: int = 6):
    dpath = DATASET_PATHS[dataset]
    legacy_train = dpath / f"train_valid_sequences_ctw_depth{depth}.csv"
    train_alias = dpath / ALIAS_FILES["train_valid_sequences.csv"]
    if train_alias.exists():
        print(f"[augment-skip] {dataset} {train_alias}", flush=True)
    elif legacy_train.exists():
        if train_alias.exists() or train_alias.is_symlink():
            train_alias.unlink()
        train_alias.symlink_to(legacy_train.name)
        print(f"[augment-link] {dataset} {train_alias} -> {legacy_train.name}", flush=True)
    else:
        print(f"[augment-run] {dataset} train_valid_sequences.csv -> {train_alias.name}", flush=True)
        cmd = [
            PYTHON,
            "examples/augment_sequences_with_ctw.py",
            "--sequence_csv",
            relpath(dpath / "train_valid_sequences.csv"),
            "--output_csv",
            relpath(train_alias),
            "--item_col",
            "questions",
            "--ctw_max_depth",
            str(depth),
            "--ctw_backend",
            "cpp",
        ]
        run_checked(cmd)
        print(f"[augment-done] {dataset} {train_alias}", flush=True)

    for source_name, alias_name in ALIAS_FILES.items():
        if source_name == "train_valid_sequences.csv":
            continue
        source = dpath / source_name
        target = dpath / alias_name
        if target.exists():
            print(f"[augment-skip] {dataset} {target}", flush=True)
            continue
        print(f"[augment-run] {dataset} {source.name} -> {target.name}", flush=True)
        cmd = [
            PYTHON,
            "examples/augment_sequences_with_ctw.py",
            "--sequence_csv",
            relpath(source),
            "--output_csv",
            relpath(target),
            "--item_col",
            "questions",
            "--support_csv",
            relpath(dpath / "train_valid_sequences.csv"),
            "--ctw_max_depth",
            str(depth),
            "--ctw_backend",
            "cpp",
        ]
        run_checked(cmd)
        print(f"[augment-done] {dataset} {target}", flush=True)


def collect_best_simplekt_jobs(dataset: str) -> list[dict]:
    patterns = [
        f"runs/**/{dataset}/simplekt/**/config.json",
        f"runs/**/simplekt/{dataset}/**/config.json",
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(str(REPO_ROOT / pattern), recursive=True))
    jobs = []
    for path_str in sorted(set(files)):
        config = read_json(Path(path_str))
        params = config["params"]
        train_config = config.get("train_config", {})
        job = {
            "dataset_name": dataset,
            "model_name": "simplekt_residual",
            "emb_type": params.get("emb_type", "qid"),
            "seed": int(params.get("seed", 42)),
            "fold": int(params["fold"]),
            "dropout": params.get("dropout", 0.1),
            "final_fc_dim": params.get("final_fc_dim", 256),
            "final_fc_dim2": params.get("final_fc_dim2", 256),
            "num_layers": params.get("num_layers", 2),
            "nheads": params.get("nheads", 4),
            "loss1": params.get("loss1", 0.5),
            "loss2": params.get("loss2", 0.5),
            "loss3": params.get("loss3", 0.5),
            "start": params.get("start", 50),
            "d_model": params.get("d_model", 64),
            "d_ff": params.get("d_ff", 256),
            "num_attn_heads": params.get("num_attn_heads", 4),
            "n_blocks": params.get("n_blocks", 4),
            "learning_rate": params.get("learning_rate", 1e-3),
            "batch_size": int(train_config.get("batch_size", 64)),
            "num_epochs": int(train_config.get("num_epochs", 50)),
            "early_stop_patience": 4,
            "ctw_feat_dim": 64,
            "ctw_delta_scale": 1.0,
            "use_wandb": 0,
            "add_uuid": 0,
        }
        jobs.append(job)
    return sorted(jobs, key=lambda row: row["fold"])


def collect_assist2012_default_jobs() -> list[dict]:
    jobs = []
    for fold in range(5):
        jobs.append(
            {
                "dataset_name": "assist2012",
                "model_name": "simplekt_residual",
                "emb_type": "qid",
                "seed": 42 if fold % 2 else 3407,
                "fold": fold,
                "dropout": 0.3,
                "final_fc_dim": 256,
                "final_fc_dim2": 64,
                "num_layers": 2,
                "nheads": 4,
                "loss1": 0.5,
                "loss2": 0.5,
                "loss3": 0.5,
                "start": 50,
                "d_model": 64,
                "d_ff": 256,
                "num_attn_heads": 4,
                "n_blocks": 4,
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 50,
                "early_stop_patience": 4,
                "ctw_feat_dim": 64,
                "ctw_delta_scale": 1.0,
                "use_wandb": 0,
                "add_uuid": 0,
            }
        )
    return jobs


def build_jobs(include_assist2012: bool) -> list[dict]:
    jobs = []
    for dataset in ["nips_task34", "algebra2005"]:
        jobs.extend(collect_best_simplekt_jobs(dataset))
    if include_assist2012:
        assist_jobs = collect_best_simplekt_jobs("assist2012")
        jobs.extend(assist_jobs if assist_jobs else collect_assist2012_default_jobs())
    for job in jobs:
        dpath = DATASET_PATHS[job["dataset_name"]]
        job["save_dir"] = str(REPO_ROOT / "runs" / "simplekt_residual_ctw" / job["dataset_name"] / "simplekt_residual")
        job["train_valid_file_override"] = relpath(dpath / ALIAS_FILES["train_valid_sequences.csv"])
        job["test_file_override"] = relpath(dpath / ALIAS_FILES["test_sequences.csv"])
        job["test_window_file_override"] = relpath(dpath / ALIAS_FILES["test_window_sequences.csv"])
        job["test_question_file_override"] = relpath(dpath / ALIAS_FILES["test_question_sequences.csv"])
        job["test_question_window_file_override"] = relpath(dpath / ALIAS_FILES["test_question_window_sequences.csv"])
    order = {"nips_task34": 0, "algebra2005": 1, "assist2012": 2}
    jobs.sort(key=lambda row: (order[row["dataset_name"]], row["fold"]))
    return jobs


def folds_token(folds) -> str:
    return ",".join(str(int(item)) for item in sorted(folds))


def cache_key(spec: dict) -> tuple[str, str, tuple[int, ...]]:
    return spec["mode"], spec["csv"], tuple(spec["folds"])


def dataset_fold_map(jobs: list[dict]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for job in jobs:
        result.setdefault(job["dataset_name"], set())
        result[job["dataset_name"]].add(int(job["fold"]))
    return {name: tuple(sorted(values)) for name, values in result.items()}


def sequence_cache_path(csv_rel: str, folds) -> Path:
    csv_path = REPO_ROOT / csv_rel
    suffix = "_" + "_".join(str(int(item)) for item in sorted(folds))
    return Path(str(csv_path) + suffix + ".pkl")


def qtest_cache_path(csv_rel: str, folds) -> Path:
    csv_path = REPO_ROOT / csv_rel
    suffix = "_" + "_".join(str(int(item)) for item in sorted(folds))
    return Path(str(csv_path) + suffix + "_qtest.pkl")


def make_cache_spec(csv_rel: str, folds, mode: str, input_type: str = "questions,concepts") -> dict:
    folds = tuple(sorted(int(item) for item in folds))
    if mode == "sequence":
        output = sequence_cache_path(csv_rel, folds)
    elif mode == "qtest":
        output = qtest_cache_path(csv_rel, folds)
    else:
        raise ValueError(f"Unsupported cache mode: {mode}")
    return {
        "mode": mode,
        "csv": csv_rel,
        "folds": folds,
        "input_type": input_type,
        "output": str(output),
    }


def required_cache_specs(job: dict, folds_by_dataset: dict[str, tuple[int, ...]], need_train: bool, need_predict: bool) -> list[dict]:
    specs: list[dict] = []
    if need_train:
        all_folds = tuple(sorted(folds_by_dataset[job["dataset_name"]]))
        train_folds = tuple(item for item in all_folds if item != int(job["fold"]))
        valid_folds = (int(job["fold"]),)
        specs.append(make_cache_spec(job["train_valid_file_override"], valid_folds, "sequence"))
        specs.append(make_cache_spec(job["train_valid_file_override"], train_folds, "sequence"))
    if need_predict:
        specs.append(make_cache_spec(job["test_file_override"], (-1,), "sequence"))
        specs.append(make_cache_spec(job["test_window_file_override"], (-1,), "sequence"))
        specs.append(make_cache_spec(job["test_question_file_override"], (-1,), "qtest"))
        specs.append(make_cache_spec(job["test_question_window_file_override"], (-1,), "qtest"))
    unique: dict[tuple[str, str, tuple[int, ...]], dict] = {}
    for spec in specs:
        unique[cache_key(spec)] = spec
    return list(unique.values())


def prebuild_script_for_mode(mode: str) -> str:
    if mode == "sequence":
        return "tools/precompute_sequence_cache_cudf.py"
    if mode == "qtest":
        return "tools/precompute_qtest_cache_cudf.py"
    raise ValueError(f"Unsupported prebuild mode: {mode}")


def build_prebuild_cmd(spec: dict, python_bin: str, device: int | str = 0) -> list[str]:
    return [
        python_bin,
        prebuild_script_for_mode(spec["mode"]),
        "--csv",
        spec["csv"],
        "--folds",
        folds_token(spec["folds"]),
        "--input-type",
        spec["input_type"],
        "--device",
        str(device),
    ]


def spec_matches_process_line(spec: dict, line: str) -> bool:
    if prebuild_script_for_mode(spec["mode"]) not in line:
        return False
    if f"--csv {spec['csv']}" not in line and f"--csv={spec['csv']}" not in line:
        return False
    token = folds_token(spec["folds"])
    if f"--folds {token}" not in line and f"--folds={token}" not in line:
        return False
    return True


def cli_args_from_job(job: dict) -> list[str]:
    args = []
    for key, value in job.items():
        args.extend([f"--{key}", str(value)])
    return args


def query_free_gpus() -> dict[int, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    result = {}
    for line in output.strip().splitlines():
        idx, total, used = [int(part.strip()) for part in line.split(",")]
        result[idx] = total - used
    return result


def launch_job(job: dict, gpu_id: int, log_dir: Path):
    train_cmd = [PYTHON, "examples/wandb_simplekt_residual_train.py", *cli_args_from_job(job)]
    ckpt = ckpt_dir(Path(job["save_dir"]), job)
    predict_cmd = [
        PYTHON,
        "examples/wandb_predict.py",
        "--bz",
        str(job["batch_size"]),
        "--save_dir",
        str(ckpt),
        "--fusion_type",
        "early_fusion,late_fusion",
        "--use_wandb",
        "0",
        "--test_file_override",
        job["test_file_override"],
        "--test_window_file_override",
        job["test_window_file_override"],
        "--test_question_file_override",
        job["test_question_file_override"],
        "--test_question_window_file_override",
        job["test_question_window_file_override"],
    ]

    env = os.environ.copy()
    repo_pythonpath = str(REPO_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        repo_pythonpath = repo_pythonpath + os.pathsep + existing_pythonpath
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONPATH": repo_pythonpath,
            "PYKT_QLEVEL_PREBUILD_MODE": "gpu",
            "PYKT_QLEVEL_PREBUILD_PYTHON": os.environ.get("PYKT_QLEVEL_PREBUILD_PYTHON", sys.executable),
            "PYKT_TORCH_COMPILE": "0",
            "PYKT_TORCH_AMP": "1",
            "PYKT_TORCH_AMP_DTYPE": "float16",
            "PYKT_PIN_MEMORY": "0",
            "PYKT_TRAIN_DATASET_DEVICE": "cuda",
            "PYKT_VALID_DATASET_DEVICE": "cuda",
            "PYKT_TEST_DATASET_DEVICE": "cuda",
            "PYKT_TRAIN_NUM_WORKERS": "0",
            "PYKT_VALID_NUM_WORKERS": "0",
            "PYKT_TEST_NUM_WORKERS": "0",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )

    log_path = log_dir / f"{job['dataset_name']}_fold{job['fold']}_gpu{gpu_id}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n=== START {time.strftime('%F %T')} gpu={gpu_id} job={job['dataset_name']} fold={job['fold']} ===\n")
    log_file.flush()
    shell = (
        "set -euo pipefail\n"
        + " ".join(shlex.quote(part) for part in train_cmd)
        + "\n"
        + " ".join(shlex.quote(part) for part in predict_cmd)
        + "\n"
    )
    proc = subprocess.Popen(
        ["bash", "-lc", shell],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return {"proc": proc, "gpu_id": gpu_id, "job": job, "log_file": log_file, "log_path": log_path}


def main():
    parser = argparse.ArgumentParser(description="Run simplekt_residual with CTW features on all datasets.")
    parser.add_argument("--include-assist2012", action="store_true")
    parser.add_argument("--min-free-mb", type=int, default=12000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()

    log_dir = REPO_ROOT / "runs" / "simplekt_residual_ctw" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["nips_task34", "algebra2005"] + (["assist2012"] if args.include_assist2012 else [])
    for dataset in datasets:
        print(f"[augment] {dataset}", flush=True)
        ensure_augmented_dataset(dataset, depth=6)

    jobs = build_jobs(include_assist2012=args.include_assist2012)
    pending = list(jobs)
    active: list[dict] = []

    while pending or active:
        still_active = []
        for state in active:
            code = state["proc"].poll()
            if code is None:
                still_active.append(state)
                continue
            state["log_file"].write(f"\n=== END {time.strftime('%F %T')} code={code} ===\n")
            state["log_file"].flush()
            state["log_file"].close()
        active = still_active

        if pending and len(active) < args.max_parallel:
            free_map = query_free_gpus()
            busy = {state["gpu_id"] for state in active}
            candidates = [
                (gpu_id, free_mb)
                for gpu_id, free_mb in sorted(free_map.items())
                if gpu_id not in busy and free_mb >= args.min_free_mb
            ]
            while pending and len(active) < args.max_parallel and candidates:
                gpu_id, _ = max(candidates, key=lambda item: item[1])
                candidates = [item for item in candidates if item[0] != gpu_id]
                job = pending.pop(0)
                state = launch_job(job, gpu_id, log_dir)
                active.append(state)
                print(
                    f"[launch] gpu={gpu_id} dataset={job['dataset_name']} fold={job['fold']} log={state['log_path']}",
                    flush=True,
                )

        if pending or active:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

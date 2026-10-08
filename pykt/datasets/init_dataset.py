import math
import os, sys
import json
import hashlib

from torch.utils.data import DataLoader
import numpy as np
import torch
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .atdkt_dataloader import ATDKTDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx
from .que_data_loader import KTQueDataset
from pykt.config import que_type_models
from .dimkt_dataloader import DIMKTDataset
from .que_data_loader_promptkt import KTQueDataset_promptKT
from .pretrain_utils import get_pretrain_data
from .scenario_utils import (
    apply_sparsity,
    get_cold_start_stats,
    filter_test_scenarios,
    build_question_holdout_set,
    apply_question_holdout_split,
    apply_question_holdout_eval,
)
import pandas as pd
import tempfile


def _env_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_str(name, default=""):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


def _move_tensor_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _move_tensor_tree(item, device)
        return value
    if isinstance(value, list):
        for idx, item in enumerate(value):
            value[idx] = _move_tensor_tree(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    return value


def _offload_dataset_tensors(dataset, device):
    for attr in ("dori", "dgaps", "dqtest"):
        if hasattr(dataset, attr):
            setattr(dataset, attr, _move_tensor_tree(getattr(dataset, attr), device))
    setattr(dataset, "_batch_fetch_device", device)
    return dataset


def _resolve_dataset_device(device_env_name, worker_env_name):
    requested = _env_str(device_env_name, "auto").strip().lower()
    if requested in {"cpu", "host"}:
        return "cpu"
    if requested in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    worker_count = _env_int(worker_env_name, _env_int("PYKT_NUM_WORKERS", 0))
    return "cuda" if worker_count == 0 else "cpu"


def _resolve_test_dataset_device():
    return _resolve_dataset_device("PYKT_TEST_DATASET_DEVICE", "PYKT_TEST_NUM_WORKERS")


def _train_dataset_cache_key(dataset_name, model_name, train_path, fold, diff_level=None):
    return (
        dataset_name,
        model_name,
        os.path.abspath(train_path),
        int(fold),
        None if diff_level is None else int(diff_level),
    )


def _build_loader(dataset, batch_size, shuffle, worker_env_name):
    num_workers = _env_int(worker_env_name, _env_int("PYKT_NUM_WORKERS", 0))
    prefetch_factor = _env_int("PYKT_PREFETCH_FACTOR", 4)
    pin_memory = os.getenv("PYKT_PIN_MEMORY", "1") != "0" and torch.cuda.is_available()
    if (
        num_workers == 0
        and getattr(dataset, "_batch_fetch_device", "cpu") != "cpu"
        and hasattr(dataset, "fetch_batch")
    ):
        return _FastBatchLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = os.getenv("PYKT_PERSISTENT_WORKERS", "1") != "0"
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


class _FastBatchLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.index_device = self._resolve_index_device()

    def _resolve_index_device(self):
        device = getattr(self.dataset, "_batch_fetch_device", "cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str) and device != "cpu" and torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")

    def __len__(self):
        return math.ceil(len(self.dataset) / max(1, self.batch_size))

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), dtype=torch.long, device=self.index_device)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long, device=self.index_device)
        for start in range(0, len(self.dataset), self.batch_size):
            yield self.dataset.fetch_batch(indices[start : start + self.batch_size])

def apply_scenario_to_path(file_path, scenario="standard", ratio=1.0, seed=42, train_path_for_cold=None):
    """
    Reads a CSV, applies scenario logic, saves to a temp file, and returns the new path.
    """
    if scenario == "standard" and ratio >= 1.0:
        return file_path
    
    df = pd.read_csv(file_path)
    if ratio < 1.0:
        df = apply_sparsity(df, ratio, seed)
    
    if scenario == "question_cold_start" and train_path_for_cold:
        train_df = pd.read_csv(train_path_for_cold)
        seen_qids = get_cold_start_stats(train_df, df)
        df = filter_test_scenarios(df, seen_qids, scenario)
    
    # Create a temp file that persists for the duration of the process
    # In a real system, we might want to cache these, but for now, temp is cleanest.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp.name, index=False)
    print(f"Scenario applied: {scenario}, ratio: {ratio}. Temp file: {tmp.name}")
    return tmp.name


def _scenario_cache_path(file_path, scenario, suffix):
    cache_dir = os.path.join(os.path.dirname(file_path), "_scenario_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(file_path)
    return os.path.join(cache_dir, f"{base}.{scenario}.{suffix}.csv")


def _question_holdout_meta_path(file_path, fold, holdout_ratio, seed):
    cache_dir = os.path.join(os.path.dirname(file_path), "_scenario_cache")
    os.makedirs(cache_dir, exist_ok=True)
    token = f"fold{fold}_ratio{holdout_ratio}_seed{seed}"
    return os.path.join(cache_dir, f"{os.path.basename(file_path)}.question_heldout.{token}.json")


def prepare_question_holdout_train_valid_path(file_path, fold, all_folds, holdout_ratio=0.25, seed=42):
    token = f"fold{fold}_ratio{holdout_ratio}_seed{seed}"
    cache_path = _scenario_cache_path(file_path, "question_heldout", token)
    meta_path = _question_holdout_meta_path(file_path, fold, holdout_ratio, seed)
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"Reusing cached question_heldout split: {cache_path}")
        return cache_path

    df = pd.read_csv(file_path)
    train_folds = set(all_folds) - {fold}
    valid_folds = {fold}
    train_df = df[df["fold"].isin(train_folds)].copy()
    heldout_qids = build_question_holdout_set(train_df, holdout_ratio=holdout_ratio, seed=seed)
    scenario_df = apply_question_holdout_split(
        df, heldout_qids=heldout_qids, train_folds=train_folds, eval_folds=valid_folds
    )
    scenario_df.to_csv(cache_path, index=False)
    with open(meta_path, "w", encoding="utf8") as fout:
        json.dump(
            {
                "scenario": "question_heldout",
                "fold": fold,
                "holdout_ratio": holdout_ratio,
                "seed": seed,
                "heldout_qids": sorted(list(heldout_qids)),
            },
            fout,
        )
    print(f"Prepared question_heldout train/valid file: {cache_path}")
    return cache_path


def prepare_question_holdout_eval_path(file_path, train_valid_path, fold, all_folds, holdout_ratio=0.25, seed=42):
    token = f"fold{fold}_ratio{holdout_ratio}_seed{seed}"
    cache_path = _scenario_cache_path(file_path, "question_heldout_eval", token)
    meta_path = _question_holdout_meta_path(train_valid_path, fold, holdout_ratio, seed)
    if os.path.exists(cache_path):
        print(f"Reusing cached question_heldout eval file: {cache_path}")
        return cache_path

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf8") as fin:
            meta = json.load(fin)
        heldout_qids = set(meta["heldout_qids"])
    else:
        train_df_all = pd.read_csv(train_valid_path)
        train_folds = set(all_folds) - {fold}
        train_df = train_df_all[train_df_all["fold"].isin(train_folds)].copy()
        heldout_qids = build_question_holdout_set(train_df, holdout_ratio=holdout_ratio, seed=seed)
        with open(meta_path, "w", encoding="utf8") as fout:
            json.dump(
                {
                    "scenario": "question_heldout",
                    "fold": fold,
                    "holdout_ratio": holdout_ratio,
                    "seed": seed,
                    "heldout_qids": sorted(list(heldout_qids)),
                },
                fout,
            )

    df = pd.read_csv(file_path)
    scenario_df = apply_question_holdout_eval(df, heldout_qids)
    scenario_df.to_csv(cache_path, index=False)
    print(f"Prepared question_heldout eval file: {cache_path}")
    return cache_path

def init_test_datasets(data_config, model_name, batch_size, diff_level=None, args=None, re_mapping=False):
    dataset_name = data_config["dataset_name"]
    # Handle Scenario for Test
    scenario = "standard"
    if args and hasattr(args, 'scenario'):
        scenario = args.scenario
    fold = 0
    if args and hasattr(args, 'fold'):
        fold = args.fold
    scenario_seed = 42
    if args and hasattr(args, 'seed'):
        scenario_seed = args.seed
    holdout_ratio = 0.25
    if args and hasattr(args, 'question_holdout_ratio'):
        holdout_ratio = args.question_holdout_ratio
    
    test_path = os.path.join(data_config["dpath"], data_config["test_file"])
    test_window_path = os.path.join(data_config["dpath"], data_config["test_window_file"])
    test_question_path = os.path.join(data_config["dpath"], data_config["test_question_file"]) if "test_question_file" in data_config else None
    test_question_window_path = os.path.join(data_config["dpath"], data_config["test_question_window_file"]) if "test_question_window_file" in data_config else None
    if args and hasattr(args, "test_file_override") and args.test_file_override:
        test_path = args.test_file_override
    if args and hasattr(args, "test_window_file_override") and args.test_window_file_override:
        test_window_path = args.test_window_file_override
    if args and hasattr(args, "test_question_file_override") and args.test_question_file_override:
        test_question_path = args.test_question_file_override
    if args and hasattr(args, "test_question_window_file_override") and args.test_question_window_file_override:
        test_question_window_path = args.test_question_window_file_override
    
    if scenario != "standard":
        train_path = os.path.join(data_config["dpath"], data_config["train_valid_file"])
        if scenario == "question_heldout":
            test_path = prepare_question_holdout_eval_path(
                test_path,
                train_path,
                fold=fold,
                all_folds=data_config["folds"],
                holdout_ratio=holdout_ratio,
                seed=scenario_seed,
            )
            test_window_path = prepare_question_holdout_eval_path(
                test_window_path,
                train_path,
                fold=fold,
                all_folds=data_config["folds"],
                holdout_ratio=holdout_ratio,
                seed=scenario_seed,
            )
            if test_question_path is not None:
                test_question_path = prepare_question_holdout_eval_path(
                    test_question_path,
                    train_path,
                    fold=fold,
                    all_folds=data_config["folds"],
                    holdout_ratio=holdout_ratio,
                    seed=scenario_seed,
                )
            if test_question_window_path is not None:
                test_question_window_path = prepare_question_holdout_eval_path(
                    test_question_window_path,
                    train_path,
                    fold=fold,
                    all_folds=data_config["folds"],
                    holdout_ratio=holdout_ratio,
                    seed=scenario_seed,
                )
        else:
            test_path = apply_scenario_to_path(test_path, scenario=scenario, train_path_for_cold=train_path)
        # Note: applying cold start to window files is trickier, but usually we evaluate on standard test for cold start.
    
    print(f"model_name is {model_name}, dataset_name is {dataset_name}")
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget", "bakt_time"]:
        test_dataset = DktForgetDataset(test_path, data_config["input_type"], {-1})
        test_window_dataset = DktForgetDataset(test_window_path,
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(test_question_path, data_config["input_type"], {-1}, True)
            test_question_window_dataset = DktForgetDataset(test_question_window_path, data_config["input_type"], {-1}, True)
    elif model_name in ["lpkt"]:
        print(f"model_name in lpkt")
        at2idx, it2idx = generate_time2idx(data_config)
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        test_dataset = KTDataset(test_path, data_config["input_type"], {-1})
        test_window_dataset = KTDataset(test_window_path, data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(test_question_path, data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(test_question_window_path, data_config["input_type"], {-1}, True)
    elif model_name in que_type_models:
        if model_name not in ["promptkt", "unikt"]:
            test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        else:
            dataset = data_config["dpath"].split("/")[-1]
            if dataset == "":
                dataset = data_config["dpath"].split("/")[-2]
            if dataset in [
                "assist2009",
                "algebra2005",
                "bridge2algebra2006",
                "nips_task34",
                "ednet",
                "peiyou",
                "ednet5w",
                "ednet_all"
            ]:
                test_dataset = KTQueDataset_promptKT(
                    os.path.join(
                        data_config["dpath"],
                        data_config["test_file_quelevel"],
                    ),
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )
                test_path_prompt = os.path.join(
                    data_config["dpath"],
                    data_config["test_window_file_quelevel"],
                )
                if not os.path.exists(test_path_prompt):
                    print("not exist")
                    sys.exit(1)
                test_window_dataset = KTQueDataset_promptKT(
                    test_path_prompt,
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )        
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["atdkt"]:
        test_dataset = ATDKTDataset(test_path, data_config["input_type"], {-1})
        test_window_dataset = ATDKTDataset(test_window_path, data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = ATDKTDataset(test_question_path, data_config["input_type"], {-1}, True)
            test_question_window_dataset = ATDKTDataset(test_question_window_path, data_config["input_type"], {-1}, True)
    elif model_name in ["dimkt"]:
        test_dataset = DIMKTDataset(data_config["dpath"], test_path, data_config["input_type"], {-1}, diff_level=diff_level)
        test_window_dataset = DIMKTDataset(data_config["dpath"], test_window_path, data_config["input_type"], {-1}, diff_level=diff_level)
        if "test_question_file" in data_config:
            test_question_dataset = DIMKTDataset(data_config["dpath"], test_question_path, data_config["input_type"], {-1}, True, diff_level=diff_level)
            test_question_window_dataset = DIMKTDataset(data_config["dpath"], test_question_window_path, data_config["input_type"], {-1}, True, diff_level=diff_level)
    elif model_name == "visual_akt":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        test_dataset = VisualKTDataset(test_path, visual_emb_path, data_config["input_type"], {-1})
        test_window_dataset = VisualKTDataset(test_window_path, visual_emb_path, data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = VisualKTDataset(test_question_path, visual_emb_path, data_config["input_type"], {-1}, qtest=True)
            test_question_window_dataset = VisualKTDataset(test_question_window_path, visual_emb_path, data_config["input_type"], {-1}, qtest=True)
    else:
        test_dataset = KTDataset(test_path, data_config["input_type"], {-1})
        test_window_dataset = KTDataset(test_window_path, data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(test_question_path, data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(test_question_window_path, data_config["input_type"], {-1}, True)

    dataset_device = _resolve_test_dataset_device()
    if dataset_device != "cpu":
        test_dataset = _offload_dataset_tensors(test_dataset, dataset_device)
        test_window_dataset = _offload_dataset_tensors(test_window_dataset, dataset_device)
        if "test_question_file" in data_config:
            if test_question_dataset is not None:
                test_question_dataset = _offload_dataset_tensors(test_question_dataset, dataset_device)
            if test_question_window_dataset is not None:
                test_question_window_dataset = _offload_dataset_tensors(test_question_window_dataset, dataset_device)

    test_loader = _build_loader(test_dataset, batch_size=batch_size, shuffle=False, worker_env_name="PYKT_TEST_NUM_WORKERS")
    test_window_loader = _build_loader(test_window_dataset, batch_size=batch_size, shuffle=False, worker_env_name="PYKT_TEST_NUM_WORKERS")
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = _build_loader(test_question_dataset, batch_size=batch_size, shuffle=False, worker_env_name="PYKT_TEST_NUM_WORKERS")
        if not test_question_window_dataset is None:
            test_question_window_loader = _build_loader(test_question_window_dataset, batch_size=batch_size, shuffle=False, worker_env_name="PYKT_TEST_NUM_WORKERS")

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None, args=None, not_select_dataset=None, re_mapping=False, dataset_cache=None):
    print(f"dataset_name:{dataset_name}")
    print(f"data_config:{data_config}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])

    # Handle Scenario for Train (Sparsity)
    train_ratio = 1.0
    if args and hasattr(args, 'train_ratio'):
        train_ratio = args.train_ratio
    scenario = "standard"
    if args and hasattr(args, 'scenario'):
        scenario = args.scenario
    scenario_seed = 42
    if args and hasattr(args, 'seed'):
        scenario_seed = args.seed
    holdout_ratio = 0.25
    if args and hasattr(args, 'question_holdout_ratio'):
        holdout_ratio = args.question_holdout_ratio
    
    train_path = os.path.join(data_config["dpath"], data_config["train_valid_file"])
    if args and hasattr(args, "train_valid_file_override") and args.train_valid_file_override:
        train_path = args.train_valid_file_override
    if scenario == "question_heldout":
        train_path = prepare_question_holdout_train_valid_path(
            train_path, fold=i, all_folds=all_folds, holdout_ratio=holdout_ratio, seed=scenario_seed
        )
    elif train_ratio < 1.0:
        train_path = apply_scenario_to_path(train_path, ratio=train_ratio)

    dataset_key = _train_dataset_cache_key(dataset_name, model_name, train_path, i, diff_level=diff_level)
    if dataset_cache is not None and dataset_key in dataset_cache:
        print(f"[dataset-cache-hit] dataset={dataset_name} fold={i} model={model_name} path={train_path}")
        curtrain, curvalid = dataset_cache[dataset_key]
        if model_name in ["dkt_forget", "bakt_time"]:
            max_rgap, max_sgap, max_pcount = 0, 0, 0
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    else:
        if model_name in ["dkt_forget", "bakt_time"]:
            max_rgap, max_sgap, max_pcount = 0, 0, 0
            curvalid = DktForgetDataset(train_path, data_config["input_type"], {i})
            curtrain = DktForgetDataset(train_path, data_config["input_type"], all_folds - {i})
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
        elif model_name == "lpkt":
            at2idx, it2idx = generate_time2idx(data_config)
            curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {i})
            curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], all_folds - {i})
        elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
            curvalid = KTDataset(train_path, data_config["input_type"], {i})
            curtrain = KTDataset(train_path, data_config["input_type"], all_folds - {i})
        elif model_name in que_type_models:
            if model_name in ["promptkt"]:
                p_dataset_name = args.dataset_name
                p_train_ratio = args.dataset_name
                if args.train_mode == "pretrain":
                    dpath = os.path.join(
                        data_config["dpath"],
                        f"train_valid_sequences_quelevel_pretrain_nomapping.csv",
                    )
                else:
                    dpath = os.path.join(
                        data_config["dpath"],
                        f"train_valid_sequences_quelevel.csv",
                    )
                print(f"train_data_path:{dpath}")
                if not os.path.exists(dpath) and args.train_mode == "pretrain":
                    print(f"loading pretrain data")
                    get_pretrain_data(data_config)
                curvalid = KTQueDataset_promptKT(
                    dpath,
                    input_type=data_config["input_type"],
                    folds={i},
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    not_select_dataset=not_select_dataset,
                    train_ratio=p_train_ratio,
                    dataset_name=p_dataset_name,
                )
                curtrain = KTQueDataset_promptKT(
                    dpath,
                    input_type=data_config["input_type"],
                    folds=all_folds - {i},
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    not_select_dataset=not_select_dataset,
                    train_ratio=p_train_ratio,
                    dataset_name=p_dataset_name,
                )
            else:
                curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
                curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        elif model_name in ["atdkt"]:
            curvalid = ATDKTDataset(train_path, data_config["input_type"], {i})
            curtrain = ATDKTDataset(train_path, data_config["input_type"], all_folds - {i})
        elif model_name == "dimkt":
            curvalid = DIMKTDataset(data_config["dpath"], train_path, data_config["input_type"], {i}, diff_level=diff_level)
            curtrain = DIMKTDataset(data_config["dpath"], train_path, data_config["input_type"], all_folds - {i}, diff_level=diff_level)
        elif model_name == "visual_akt":
            visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
            curvalid = VisualKTDataset(train_path, visual_emb_path, data_config["input_type"], {i})
            curtrain = VisualKTDataset(train_path, visual_emb_path, data_config["input_type"], all_folds - {i})
        else:
            curvalid = KTDataset(train_path, data_config["input_type"], {i})
            curtrain = KTDataset(train_path, data_config["input_type"], all_folds - {i})
        train_dataset_device = _resolve_dataset_device("PYKT_TRAIN_DATASET_DEVICE", "PYKT_TRAIN_NUM_WORKERS")
        valid_dataset_device = _resolve_dataset_device("PYKT_VALID_DATASET_DEVICE", "PYKT_VALID_NUM_WORKERS")
        if train_dataset_device != "cpu":
            curtrain = _offload_dataset_tensors(curtrain, train_dataset_device)
        if valid_dataset_device != "cpu":
            curvalid = _offload_dataset_tensors(curvalid, valid_dataset_device)
        if dataset_cache is not None:
            dataset_cache[dataset_key] = (curtrain, curvalid)

    train_loader = _build_loader(curtrain, batch_size=batch_size, shuffle=True, worker_env_name="PYKT_TRAIN_NUM_WORKERS")
    valid_loader = _build_loader(curvalid, batch_size=batch_size, shuffle=False, worker_env_name="PYKT_VALID_NUM_WORKERS")
    
    try:
        if model_name in ["dkt_forget", "bakt_time"]:
            test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
            # test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
            #                                 data_config["input_type"], {-1})
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
    #     elif model_name == "lpkt":
    #         test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #         # test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #     elif model_name in que_type_models:
    #         test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
    #                         input_type=data_config["input_type"], folds=[-1], 
    #                         concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    #     else:
    #         test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
    #         # test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    except:
        pass
    
    if model_name in ["dkt_forget", "bakt_time"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
    if model_name == "lpkt":
        print(f"num_at:{len(at2idx)}")
        print(f"num_it:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = None
    return train_loader, valid_loader#, test_loader, test_window_loader

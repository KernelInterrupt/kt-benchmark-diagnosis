import os
import sys
import argparse
import json
import copy
import torch
import pandas as pd
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.models import evaluate,evaluate_question,load_model
from pykt.datasets import init_test_datasets
from tools.per_question_common import infer_loader_family

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def _build_args_obj(params, fold, trained_params):
    args_obj = params.get('args')
    if args_obj is None:
        args_obj = SimpleNamespace()
    if not hasattr(args_obj, "fold"):
        args_obj.fold = fold
    if not hasattr(args_obj, "seed"):
        args_obj.seed = trained_params.get("seed", 42)
    if not hasattr(args_obj, "question_holdout_ratio"):
        args_obj.question_holdout_ratio = trained_params.get("question_holdout_ratio", 0.25)
    if not hasattr(args_obj, "scenario"):
        args_obj.scenario = params.get("scenario", "standard")
    for name in [
        "test_file_override",
        "test_window_file_override",
        "test_question_file_override",
        "test_question_window_file_override",
        "dataset_name",
    ]:
        if not hasattr(args_obj, name):
            setattr(args_obj, name, params.get(name, ""))
    return args_obj


def _load_job_spec(params):
    smoke_mode = os.getenv("PYKT_SMOKE_TEST", "0") == "1"
    save_dir = params["save_dir"]
    batch_size = params["bz"]
    fusion_type = params["fusion_type"].split(",")

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in [
            'use_wandb','learning_rate','add_uuid','l2','selection_metric',
            # Dataset/evaluation controls; these are consumed through args_obj
            # and must not be passed into vanilla model constructors.
            'scenario','train_ratio','question_holdout_ratio',
            'train_valid_file_override','test_file_override','test_window_file_override',
            'test_question_file_override','test_question_window_file_override',
            'predict_batch_size','include_qtest_eval','ratio_tag','method','variant',
            'result_json','wandb_group','wandb_job_type','wandb_name','project_name',
            'cold_protocol',
        ]:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        train_dataset_name = trained_params["dataset_name"]
        model_name = trained_params["model_name"]
        dataset_name = train_dataset_name
        emb_type = trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    if params.get("dataset_name", ""):
        dataset_name = params["dataset_name"]

    with open(os.path.join(CONFIG_DIR, "data_config.json")) as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if "dpath" in data_config:
            data_config["dpath"] = os.path.normpath(os.path.join(EXAMPLES_DIR, data_config["dpath"]))
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]

    args_obj = _build_args_obj(params, fold, trained_params)
    if int(params.get("include_qtest_eval", 0)) == 0:
        data_config.pop("test_question_file", None)
        data_config.pop("test_question_window_file", None)

    ignore_keys = []
    is_cross_dataset = dataset_name != train_dataset_name
    if is_cross_dataset:
        if params.get("cross_dataset_ignore_visual", 0) == 1:
            ignore_keys.append("visual_emb.weight")
        if params.get("cross_dataset_ignore_text", 0) == 1:
            ignore_keys.append("text_emb.weight")
        if params.get("cross_dataset_ignore_frozen_embeddings", 0) == 1:
            if emb_type == "visual_only":
                ignore_keys.append("visual_emb.weight")
            elif emb_type == "text_only":
                ignore_keys.append("text_emb.weight")

    return {
        "smoke_mode": smoke_mode,
        "save_dir": save_dir,
        "batch_size": batch_size,
        "fusion_type": fusion_type,
        "config": config,
        "model_config": model_config,
        "trained_params": trained_params,
        "fold": fold,
        "train_dataset_name": train_dataset_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "emb_type": emb_type,
        "data_config": data_config,
        "args_obj": args_obj,
        "ignore_keys": sorted(set(ignore_keys)),
        "diff_level": trained_params.get("difficult_levels"),
        "save_predictions": int(params.get("save_predictions", 1)),
    }


def _dataset_cache_key(job):
    args_obj = job["args_obj"]
    loader_family = infer_loader_family(job["model_name"], job["dataset_name"])
    return (
        job["dataset_name"],
        loader_family,
        job["batch_size"],
        job["diff_level"],
        getattr(args_obj, "scenario", "standard"),
        getattr(args_obj, "fold", job["fold"]),
        getattr(args_obj, "seed", job["trained_params"].get("seed", 42)),
        getattr(args_obj, "question_holdout_ratio", job["trained_params"].get("question_holdout_ratio", 0.25)),
        getattr(args_obj, "test_file_override", ""),
        getattr(args_obj, "test_window_file_override", ""),
        getattr(args_obj, "test_question_file_override", ""),
        getattr(args_obj, "test_question_window_file_override", ""),
    )


def _get_test_loaders(job, dataset_cache=None):
    cache_key = _dataset_cache_key(job)
    if dataset_cache is not None and cache_key in dataset_cache:
        return dataset_cache[cache_key]

    if job["model_name"] != "dimkt":
        loaders = init_test_datasets(job["data_config"], job["model_name"], job["batch_size"], args=job["args_obj"])
    else:
        loaders = init_test_datasets(
            job["data_config"],
            job["model_name"],
            job["batch_size"],
            diff_level=job["diff_level"],
            args=job["args_obj"],
        )
    if dataset_cache is not None:
        dataset_cache[cache_key] = loaders
    return loaders


def _describe_loader(name, loader):
    if loader is None:
        print(f"{name}: None")
        return
    dataset = getattr(loader, "dataset", None)
    dataset_name = type(dataset).__name__ if dataset is not None else "None"
    loader_name = type(loader).__name__
    batch_fetch_device = getattr(dataset, "_batch_fetch_device", "cpu") if dataset is not None else "cpu"
    has_fetch_batch = hasattr(dataset, "fetch_batch") if dataset is not None else False
    print(
        f"{name}: loader={loader_name}, dataset={dataset_name}, "
        f"batch_fetch_device={batch_fetch_device}, has_fetch_batch={has_fetch_batch}"
    )


def _evaluate_loaded_model(job, model, test_loaders):
    smoke_mode = job["smoke_mode"]
    model_name = job["model_name"]
    save_dir = job["save_dir"]
    data_config = job["data_config"]
    fusion_type = job["fusion_type"]
    emb_type = job["emb_type"]
    save_predictions = int(job.get("save_predictions", 1))
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = test_loaders

    should_save = (not smoke_mode) and save_predictions == 1
    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt") if should_save else ""

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {job["fold"]}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
    else:
        rel = None

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_path=save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt") if should_save else ""
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_path=save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }

    if "test_question_file" in data_config and test_question_loader is not None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt") if should_save else ""
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]

    if "test_question_window_file" in data_config and test_question_window_loader is not None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt") if should_save else ""
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

    print(dres)
    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])
    return dres


def run_prediction_job(params, dataset_cache=None):
    job = _load_job_spec(params)
    test_loaders = _get_test_loaders(job, dataset_cache=dataset_cache)
    _describe_loader("test_loader", test_loaders[0])
    _describe_loader("test_window_loader", test_loaders[1])
    _describe_loader("test_question_loader", test_loaders[2])
    _describe_loader("test_question_window_loader", test_loaders[3])

    print(f"Start predicting model: {job['model_name']}, embtype: {job['emb_type']}, save_dir: {job['save_dir']}, dataset_name: {job['dataset_name']}")
    print(f"model_config: {job['model_config']}")
    print(f"data_config: {job['data_config']}")

    model = load_model(
        job["model_name"],
        job["model_config"],
        job["data_config"],
        job["emb_type"],
        job["save_dir"],
        strict=(len(job["ignore_keys"]) == 0),
        ignore_keys=job["ignore_keys"],
    )
    return _evaluate_loaded_model(job, model, test_loaders)


def main(params):
    smoke_mode = os.getenv("PYKT_SMOKE_TEST", "0") == "1"
    if params['use_wandb'] ==1:
        import wandb
        with open(os.path.join(CONFIG_DIR, "wandb.json")) as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    dres = run_prediction_job(params)

    if params['use_wandb'] ==1:
        wandb.log(dres)
    return dres

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="", help="Optional target dataset override for cross-dataset evaluation")
    parser.add_argument("--cross_dataset_ignore_visual", type=int, default=0, help="If 1, ignore visual_emb.weight when loading checkpoint")
    parser.add_argument("--cross_dataset_ignore_text", type=int, default=0, help="If 1, ignore text_emb.weight when loading checkpoint")
    parser.add_argument("--cross_dataset_ignore_frozen_embeddings", type=int, default=0, help="If 1, automatically ignore frozen item embeddings (visual/text) when target dataset differs from training dataset")
    parser.add_argument("--scenario", type=str, default="standard", choices=["standard", "question_cold_start", "question_heldout"], help="Evaluation scenario")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Fraction of training data to use (Sparsity experiment)")
    parser.add_argument("--question_holdout_ratio", type=float, default=0.25, help="Held-out question ratio for the question_heldout scenario")
    parser.add_argument("--test_file_override", type=str, default="")
    parser.add_argument("--test_window_file_override", type=str, default="")
    parser.add_argument("--test_question_file_override", type=str, default="")
    parser.add_argument("--test_question_window_file_override", type=str, default="")
    parser.add_argument("--save_predictions", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    params['args'] = args # Store args object in params for init_test_datasets
    main(params)

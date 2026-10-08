import os
import argparse
import json
import hashlib
from collections import OrderedDict

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model
from pykt.models.train_model import get_last_train_summary
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime

if os.environ.get("PYKT_CUDA_LAUNCH_BLOCKING", "").strip().lower() in {"1", "true", "on", "yes"}:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
device = "cpu" if not torch.cuda.is_available() else "cuda"
if os.environ.get("PYKT_DETERMINISTIC", "").strip().lower() in {"1", "true", "on", "yes"}:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
TORCH_COMPILE_WHITELIST = {
    "akt",
    "akt_residual",
    "deep_irt",
    "dkt",
    "dkt+",
    "dkvmn",
    "extrakt",
    "folibikt",
    "hawkes",
    "saint",
    "sakt",
}
TORCH_AMP_WHITELIST = {
    "akt",
    "akt_residual",
    "atdkt",
    "atkt",
    "dkt",
    "dkt_residual",
    "dkt+",
    "extrakt",
    "folibikt",
    "qdkt",
    "qikt",
    "saint",
}


def maybe_compile_model(model_name, model):
    compile_policy = os.environ.get("PYKT_TORCH_COMPILE", "0").strip().lower()
    if compile_policy in {"0", "false", "off", "no", "disable", "disabled"}:
        return model
    if not hasattr(torch, "compile"):
        return model
    if compile_policy == "auto" and model_name not in TORCH_COMPILE_WHITELIST:
        return model

    compile_target = model.model if hasattr(model, "model") else model
    compile_mode = os.environ.get("PYKT_TORCH_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    try:
        compiled_target = torch.compile(compile_target, mode=compile_mode)
    except Exception as exc:
        print(f"torch.compile skipped for {model_name}: {exc}")
        return model

    if hasattr(model, "model"):
        model.model = compiled_target
    else:
        model = compiled_target
    print(f"torch.compile enabled for {model_name} with mode={compile_mode}")
    return model


def resolve_amp_settings(model_name):
    amp_policy = os.environ.get("PYKT_TORCH_AMP", "auto").strip().lower()
    disabled = {"0", "false", "off", "no", "disable", "disabled"}
    enabled = {"1", "true", "on", "yes", "enable", "enabled"}
    if not torch.cuda.is_available() or amp_policy in disabled:
        return False, None
    if amp_policy == "auto" and model_name not in TORCH_AMP_WHITELIST:
        return False, None
    if amp_policy not in enabled and amp_policy != "auto":
        return False, None

    amp_dtype_name = os.environ.get("PYKT_TORCH_AMP_DTYPE", "float16").strip().lower()
    amp_dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    amp_dtype = amp_dtypes.get(amp_dtype_name)
    if amp_dtype is None:
        print(f"AMP skipped for {model_name}: unsupported dtype {amp_dtype_name}")
        return False, None

    print(f"AMP enabled for {model_name} with dtype={amp_dtype_name}")
    return True, amp_dtype

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)


def normalize_loaded_state_dict(state_dict):
    return OrderedDict(
        (key.replace("._orig_mod.", ".").replace("_orig_mod.", ""), value)
        for key, value in state_dict.items()
    )

def main(params, dataset_cache=None):
    if "use_wandb" not in params:
        params['use_wandb'] = 1
    wandb = None

    if params['use_wandb']==1:
        import wandb
        wandb_kwargs = {}
        if params.get("project_name"):
            wandb_kwargs["project"] = params["project_name"]
        if params.get("wandb_group"):
            wandb_kwargs["group"] = params["wandb_group"]
        if params.get("wandb_job_type"):
            wandb_kwargs["job_type"] = params["wandb_job_type"]
        if params.get("wandb_name"):
            wandb_kwargs["name"] = params["wandb_name"]
        wandb.init(config=params, **wandb_kwargs)

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open(os.path.join(CONFIG_DIR, "kt_config.json")) as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "akt_residual", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt", "gmlp", "visual_akt", "visual_ema", "visual_dkt", "visual_attn", "visual_rwkv", "visual_gema", "visual_abqr", "visual_irt", "visual_gmlp", "visual_gmlp_nocf", "visual_gmlp_causal", "visual_gmlp_nocf_causal", "visual_gmlp_nocf_causal_gate", "visual_gmlp_nocf_causal_gate_adapter", "text_gmlp_nocf_causal_gate", "text_gmlp_nocf_causal_gate_adapter"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt","simplekt_residual","dkt_residual","stablekt", "datakt", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        for key in ["batch_size", "num_epochs", "early_stop_patience"]:
            if key in model_config:
                del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = min(params['num_epochs'], 50)
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open(os.path.join(CONFIG_DIR, "data_config.json")) as fin:
        data_config = json.load(fin)
    for dataset_key, cfg in data_config.items():
        if isinstance(cfg, dict) and "dpath" in cfg:
            cfg["dpath"] = os.path.normpath(os.path.join(EXAMPLES_DIR, cfg["dpath"]))
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")
    from types import SimpleNamespace
    args_obj = SimpleNamespace(**params)
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, args=args_obj, dataset_cache=dataset_cache)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level, args=args_obj, dataset_cache=dataset_cache)

    # Sanitize parameter values to avoid directory creation issues (e.g. "google/siglip" -> "google_siglip")
    params_str = "_".join([str(v).replace("/", "_") for k,v in params.items() if not k in ['other_config']])
    if len(params_str) > 180:
        params_hash = hashlib.sha1(params_str.encode("utf-8")).hexdigest()[:16]
        params_str = params_str[:120] + "__" + params_hash

    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str+f"_{ str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    for remove_item in [
        'use_wandb','learning_rate','add_uuid','l2','selection_metric',
        # Dataset/evaluation controls; these are consumed by init_dataset4train
        # via args_obj and must not be passed into vanilla model constructors.
        'scenario','train_ratio','question_holdout_ratio',
        'train_valid_file_override','test_file_override','test_window_file_override',
        'test_question_file_override','test_question_window_file_override',
        'predict_batch_size','include_qtest_eval','ratio_tag','method','variant',
        'result_json','wandb_group','wandb_job_type','wandb_name','project_name',
        'cold_protocol',
    ]:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","simplekt_residual","akt_residual","stablekt", "datakt","folibikt"]:
        model_config["seq_len"] = seq_len
        
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, copy.deepcopy(model_config), data_config[dataset_name], emb_type)
    model = maybe_compile_model(model_name, model)
    amp_enabled, amp_dtype = resolve_amp_settings(model_name)
    print(f"model is {model}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print(f"dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    early_stop_patience = int(params.get("early_stop_patience", 4))
    selection_metric = params.get("selection_metric", "validauc")
    
    debug_print(text = "train model",fuc_name="main")
    
    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold, early_stop_patience=early_stop_patience, amp_enabled=amp_enabled, amp_dtype=amp_dtype, selection_metric=selection_metric)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, early_stop_patience=early_stop_patience, amp_enabled=amp_enabled, amp_dtype=amp_dtype, selection_metric=selection_metric)
    
    if save_model:
        best_model = init_model(model_name, copy.deepcopy(model_config), data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
        net = normalize_loaded_state_dict(net)
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    print(f"end:{datetime.datetime.now()}")
    train_summary = get_last_train_summary()
    
    if params['use_wandb']==1:
        log_payload = {
            "validauc": validauc,
            "validacc": validacc,
            "best_epoch": best_epoch,
            "model_save_path": model_save_path,
        }
        log_payload.update(train_summary)
        wandb.log(log_payload)
        wandb.finish()
    return {
        "testauc": testauc,
        "testacc": testacc,
        "window_testauc": window_testauc,
        "window_testacc": window_testacc,
        "validauc": validauc,
        "validacc": validacc,
        "best_epoch": best_epoch,
        "model_save_path": model_save_path,
        **train_summary,
    }

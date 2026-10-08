#!/usr/bin/env python3

QDATASET_MODELS = {
    "iekt",
    "qdkt",
    "qikt",
    "lpkt",
    "rkt",
    "promptkt",
    "unikt",
    "qikt_ab_a+b+c",
    "qikt_ab_a+b+c+irt",
    "qikt_ab_a+b+irt",
    "qikt_ab_a+c+irt",
    "qikt_ab_a+irt",
    "qikt_ab_b+irt",
}

RKT_KT_DATASETS = {
    "statics2011",
    "assist2015",
    "poj",
}


def infer_loader_family(model_name: str, dataset_name: str = "") -> str:
    model_name = (model_name or "").strip()
    dataset_name = (dataset_name or "").strip()

    if model_name in {"dkt_forget", "bakt_time"}:
        return "dkt_forget"
    if model_name == "atdkt":
        return "atdkt"
    if model_name == "dimkt":
        return "dimkt"
    if model_name == "qdkt":
        return "qdkt"
    if model_name == "lpkt":
        return "lpkt"
    if model_name == "rkt":
        return "kt" if dataset_name in RKT_KT_DATASETS else "que"
    if model_name in {"promptkt", "unikt"}:
        return "que_prompt"
    if model_name in QDATASET_MODELS:
        return "que"
    return "kt"


def resident_family_key(dataset_name: str, model_name: str) -> str:
    dataset_name = (dataset_name or "").strip()
    return f"{dataset_name}::{infer_loader_family(model_name, dataset_name)}"

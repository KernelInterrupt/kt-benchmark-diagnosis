import os, sys
import threading
from collections import OrderedDict
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.amp import GradScaler
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TRAIN_SUMMARY_LOCAL = threading.local()


def normalized_state_dict(model):
    state = model.state_dict()
    cleaned = OrderedDict()
    for key, value in state.items():
        cleaned[key.replace("._orig_mod.", ".").replace("_orig_mod.", "")] = value
    return cleaned


def binary_cross_entropy_fp32(pred, target):
    device_type = pred.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return binary_cross_entropy(pred.float(), target.float())


def _normalize_selection_metric(metric_name):
    name = str(metric_name or "validauc").strip().lower().replace("_", "")
    if name.startswith("valid"):
        name = name[5:]
    aliases = {
        "auc": "auc",
        "acc": "acc",
        "nll": "nll",
        "brier": "brier",
        "ece": "ece",
        "logloss": "nll",
    }
    return aliases.get(name, "auc")


def _selection_mode(metric_name):
    return "min" if metric_name in {"nll", "brier", "ece"} else "max"


def _is_metric_improved(current_value, best_value, metric_name):
    if np.isnan(current_value):
        return False
    mode = _selection_mode(metric_name)
    tol = 1e-4 if mode == "min" else 1e-3
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value - tol
    return current_value > best_value + tol


def get_last_train_summary():
    return dict(getattr(_TRAIN_SUMMARY_LOCAL, "value", {}))

def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    if model_name in ["visual_ema", "atdkt", "simplekt", "simplekt_residual", "dkt_residual", "akt_residual", "irt", "stablekt", "datakt", "sparsekt", "cskt", "hcgkt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy_fp32(y, t)

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
    elif model_name in ["rekt"]:
        # print("ys shape:", ys[0].shape)
        # print("sm shape:", sm.shape)
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy_fp32(y, t)
    
    elif model_name in ["ukt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = binary_cross_entropy_fp32(y, t)
        if model.use_CL:
            loss2 = ys[1]
            loss1 = loss1 + model.cl_weight * loss2
        loss =loss1

    elif model_name in ["rkt","dimkt","dkt", "gmlp", "visual_dkt", "visual_ema", "visual_attn", "visual_rwkv", "visual_gema", "visual_abqr", "visual_irt", "visual_gmlp", "visual_gmlp_nocf", "visual_gmlp_causal", "visual_gmlp_nocf", "visual_gmlp_causal", "visual_gmlp_nocf_causal", "visual_gmlp_nocf_causal_gate", "visual_gmlp_nocf_causal_gate_adapter", "text_gmlp_nocf_causal_gate", "text_gmlp_nocf_causal_gate_adapter", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy_fp32(y, t)
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy_fp32(y_next, r_next)

        loss_r = binary_cross_entropy_fp32(y_curr, r_curr) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt","akt_residual","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","lefokt_akt", "dtransformer", "fluckt", "visual_akt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy_fp32(y, t) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy_fp32(y, t) * y.numel()
    
    return loss


def model_forward(model, data, rel=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "datakt"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "simplekt_residual", "irt", "stablekt", "cskt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["dkt_residual"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y, y2, y3]
    elif model_name in ["akt_residual"]:
        y, c_reg_loss, _ = model(dcur, train=True)
        ys = [y[:,1:], 0, 0]
        if c_reg_loss != 0:
            preloss.append(c_reg_loss)
    elif model_name in ["sparsekt"]:
        model_outputs = model(dcur, train=True)
        if len(model_outputs) == 4:
            y, y2, y3, c_reg_loss = model_outputs
            if c_reg_loss != 0:
                preloss.append(c_reg_loss)
        else:
            y, y2, y3 = model_outputs
        ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["ukt"]:
        if model.use_CL != 0 :
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:,1:],sim,y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:,1:], y2, y3]
    elif model_name in ["hcgkt"]:
        dataset_name = data_config["dpath"].split("/")[-1]
        emb_size = model.emb_size
        step_size = model.step_size
        step_m = model.step_m
        grad_clip = model.grad_clip
        mm = model.mm

        # the xxx.pt file of pre_load_gcn can be found in :
        # https://drive.google.com/drive/folders/1JWstsquI3TzbUlqB1EyCbjem4qPyRLCh?usp=drive_link
        matrix = None
        if dataset_name == 'assist2009':
            pre_load_gcn = "../data/assist2009/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'algebra2005':
            pre_load_gcn = "../data/algebra2005/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'bridge2algebra2006':
            pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'peiyou':
            pre_load_gcn = "../data/peiyou/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'nips_task34':
            pre_load_gcn = "../data/nips_task34/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        perturb_shape = (matrix.shape[0], emb_size)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
        perturb.requires_grad_()
        y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
        ys = [y[:,1:], y2, y3]
        loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        loss /= step_m
        opt.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
            ys = [y[:,1:], y2, y3]
            loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
            loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.sfm_cl.gcl.update_target_network(mm)  
        return loss
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["datakt"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1).to(device)
    if model_name == "visual_abqr":
        if model.training:
            # 1. Forward to get online representation and KT prediction
            y, online_z, vis_feat = model(q.long(), r.long())
            y_proj = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            y_sel = torch.masked_select(y_proj, sm)
            t_sel = torch.masked_select(rshft, sm)
            from torch.nn.functional import binary_cross_entropy, mse_loss
            loss_kt = binary_cross_entropy(y_sel.double(), t_sel.double())
            
            # Target representation (initially unperturbed)
            v_q_target = model.get_target_v(vis_feat)
            
            # Contrastive Loss
            import torch.nn.functional as F
            online_z_norm = F.normalize(online_z, p=2, dim=-1)
            v_q_target_norm = F.normalize(v_q_target, p=2, dim=-1)
            loss_cl = 2 - 2 * (online_z_norm * v_q_target_norm).sum(dim=-1).mean()
            
            # Adversarial target perturbation
            total_loss = loss_kt + model.alpha * loss_cl
            
            features_grad = grad(total_loss, vis_feat, retain_graph=True)[0]
            
            # FGM: Create adversarial perturbation
            p_adv = model.epsilon * torch.sign(features_grad)
            
            # Compute Target with perturbation
            v_q_target_adv = model.get_target_v(vis_feat + p_adv.detach())
            v_q_target_adv_norm = F.normalize(v_q_target_adv, p=2, dim=-1)
            
            # Final Contrastive Loss with hard target
            loss_cl_adv = 2 - 2 * (online_z_norm * v_q_target_adv_norm).sum(dim=-1).mean()
            
            # The final loss to optimize
            loss = loss_kt + model.alpha * loss_cl_adv
            
            # Update target network (EMA)
            model.update_target_network()
            
            return loss # bypasses cal_loss call at the end
        else:
            y = model(q.long(), r.long())
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)
    if model_name in ["gmlp", "visual_gmlp", "visual_gmlp_nocf", "visual_gmlp_causal", "visual_gmlp_nocf_causal", "visual_gmlp_nocf_causal_gate", "visual_gmlp_nocf_causal_gate_adapter", "text_gmlp_nocf_causal_gate", "text_gmlp_nocf_causal_gate_adapter"]:
        # Standard GMLP and VisualGMLP both use target-aware prediction
        if model_name == "gmlp" and model.num_q > 0:
            y = model(q.long(), r.long(), qshft.long(), c=c.long(), cshft=cshft.long()) # Question level
        elif model_name == "gmlp":
            y = model(c.long(), r.long(), cshft.long()) # Concept level
        elif model_name in ["visual_gmlp_nocf", "visual_gmlp_nocf_causal", "visual_gmlp_nocf_causal_gate", "visual_gmlp_nocf_causal_gate_adapter", "text_gmlp_nocf_causal_gate", "text_gmlp_nocf_causal_gate_adapter"]:
            y = model(q.long(), r.long(), qshft.long())
        else:
            y = model(q.long(), r.long(), qshft.long(), c=c.long(), cshft=cshft.long()) # VisualGMLP with concept fusion
        ys.append(y)
    elif model_name in ["visual_dkt", "visual_ema", "visual_attn", "visual_rwkv", "visual_gema", "visual_irt"]:
        y = model(q.long(), r.long())
        if model_name != "visual_ema":
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lefokt_akt", "fluckt"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name == "visual_akt":
        v, r, vshft, rshft = dcur["vseqs"].to(device), dcur["rseqs"].to(device), dcur["shft_vseqs"].to(device), dcur["shft_rseqs"].to(device)
        # Construct full sequences of length L
        cv = torch.cat((v[:, 0:1], vshft), dim=1)
        cr = torch.cat((r[:, 0:1], rshft), dim=1)
        y, reg_loss = model(cv, cr)
        ys.append(y[:, 1:]) # Slice to match rshft (length L-1)
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss,temp
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None, early_stop_patience=10, amp_enabled=False, amp_dtype=None, selection_metric="validauc"):
    best_epoch = -1
    best_metric_name = _normalize_selection_metric(selection_metric)
    best_metric_value = None
    best_valid_metrics = {
        "auc": float("nan"),
        "acc": float("nan"),
        "nll": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }
    train_step = 0
    amp_enabled = bool(amp_enabled and device.type == "cuda" and amp_dtype is not None)
    scaler = GradScaler("cuda", enabled=amp_enabled)
    smoke_max_batches = 0
    try:
        smoke_max_batches = int(os.getenv("PYKT_SMOKE_MAX_BATCHES", "0") or "0")
    except ValueError:
        smoke_max_batches = 0
    if smoke_max_batches > 0:
        num_epochs = min(int(num_epochs), 1)

    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = float("nan"), float("nan")
    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        num_train_batches = len(train_loader)
        log_every = max(1, min(200, num_train_batches // 10 if num_train_batches > 10 else 1))
        print(f"Epoch {i}/{num_epochs} started: train_batches={num_train_batches}, log_every={log_every}, model={model.model_name}, emb_type={model.emb_type}")
        for batch_idx, data in enumerate(train_loader, start=1):
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            opt.zero_grad()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                if model.model_name=='rkt':
                    loss = model_forward(model, data, rel)
                elif model.model_name in ["ukt"] and model.use_CL != 0:
                    loss,temp = model_forward(model, data)
                else:
                    loss = model_forward(model, data)
            if amp_enabled:
                scaler.scale(loss).backward()
                if model.model_name == "rkt":
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), model.grad_clip)
                if model.model_name == "dtransformer":
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if model.model_name in ["atkt", "atktfix"]:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()#compute gradients
                if model.model_name == "rkt":
                    clip_grad_norm_(model.parameters(), model.grad_clip)
                if model.model_name == "dtransformer":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if model.model_name in ["atkt", "atktfix"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_train_batches:
                running_loss = float(np.mean(loss_mean))
                print(
                    f"Epoch {i}/{num_epochs} | batch {batch_idx}/{num_train_batches} "
                    f"| global_step {train_step} | running_loss {running_loss:.6f}"
                )
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
            if smoke_max_batches > 0 and batch_idx >= smoke_max_batches:
                break
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        if model.model_name=='rkt':
            eval_result = evaluate(model, valid_loader, model.model_name, rel, return_details=True)
        else:
            eval_result = evaluate(model, valid_loader, model.model_name, return_details=True)
        auc = eval_result["auc"]
        acc = eval_result["acc"]
        metric_value = eval_result[best_metric_name]
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if _is_metric_improved(metric_value, best_metric_value, best_metric_name):
            if save_model:
                torch.save(normalized_state_dict(model), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            best_metric_value = metric_value
            best_valid_metrics = dict(eval_result)
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(
            f"Epoch: {i}, validauc: {auc:.4}, validacc: {acc:.4}, validnll: {eval_result['nll']:.6f}, "
            f"validbrier: {eval_result['brier']:.6f}, validece: {eval_result['ece']:.6f}, "
            f"best epoch: {best_epoch}, best {best_metric_name}: {best_metric_value:.6f}, "
            f"train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}"
        )
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= early_stop_patience:
            break
    validauc = best_valid_metrics["auc"]
    validacc = best_valid_metrics["acc"]
    _TRAIN_SUMMARY_LOCAL.value = {
        "best_epoch": best_epoch,
        "selection_metric": f"valid{best_metric_name}",
        "selection_mode": _selection_mode(best_metric_name),
        "selection_value": best_metric_value,
        "validauc": best_valid_metrics["auc"],
        "validacc": best_valid_metrics["acc"],
        "validnll": best_valid_metrics["nll"],
        "validbrier": best_valid_metrics["brier"],
        "validece": best_valid_metrics["ece"],
    }
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch

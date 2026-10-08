import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_residual import DKTResidual
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .irt import IRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .akt_residual import AKTResidual
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .simplekt_residual import SimpleKTResidual
from .datakt import BAKTTime
from .qdkt import QDKT
from .dimkt import DIMKT
from .sparsekt import sparseKT
from .rkt import RKT
from .folibikt import folibiKT
from .dtransformer import DTransformer
from .stablekt import stableKT
from .extrakt import extraKT
from .rekt import ReKT
from .cskt import CSKT
from .lefokt_akt import LEFOKT_AKT
from .ukt import UKT
from .hcgkt import HCGKT
from .robustkt import Robustkt
from .gmlp import GMLP

device = "cpu" if not torch.cuda.is_available() else "cuda"


def _require_lpkt_symbols():
    from .lpkt import LPKT
    from .lpkt_utils import generate_qmatrix

    return LPKT, generate_qmatrix


def _require_qikt_symbol():
    from .qikt import QIKT

    return QIKT


def _infer_num_users(data_config):
    if "num_u" in data_config and data_config["num_u"] is not None:
        return int(data_config["num_u"])
    override = data_config.get("train_valid_file_override", "")
    if override:
        train_path = override
    else:
        train_path = os.path.join(data_config["dpath"], data_config["train_valid_file"])
    import pandas as pd

    df = pd.read_csv(train_path, usecols=["uid"])
    return int(df["uid"].max()) + 1

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_residual":
        model = DKTResidual(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "gmlp":
        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
        
        model = GMLP(num_c=data_config["num_c"], num_q=data_config.get("num_q", 0), 
                     emb_size=emb_size, emb_type=emb_type, **model_config).to(device)
    elif model_name == "visual_gmlp":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualGMLP(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_gmlp_nocf":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]

        model = VisualGMLPNoCF(num_c=data_config["num_c"], emb_size=emb_size,
                               visual_embedding=visual_embedding, emb_type=emb_type,
                               **model_config).to(device)
    elif model_name == "visual_gmlp_causal":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]

        model = VisualGMLPCausal(num_c=data_config["num_c"], emb_size=emb_size,
                                 visual_embedding=visual_embedding, emb_type=emb_type,
                                 **model_config).to(device)
    elif model_name == "visual_gmlp_nocf_causal":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]

        model = VisualGMLPNoCFCausal(num_c=data_config["num_c"], emb_size=emb_size,
                                     visual_embedding=visual_embedding, emb_type=emb_type,
                                     **model_config).to(device)
    elif model_name == "visual_gmlp_nocf_causal_gate":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config:
            del model_config["emb_size"]

        model = VisualGMLPNoCFCausalGate(num_c=data_config["num_c"], emb_size=emb_size,
                                         visual_embedding=visual_embedding, emb_type=emb_type,
                                         **model_config).to(device)
    elif model_name == "visual_gmlp_nocf_causal_gate_adapter":
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config:
            del model_config["emb_size"]

        model = VisualGMLPNoCFCausalGateAdapter(
            num_c=data_config["num_c"],
            emb_size=emb_size,
            visual_embedding=visual_embedding,
            emb_type=emb_type,
            **model_config
        ).to(device)
    elif model_name == "text_gmlp_nocf_causal_gate":
        text_emb_path = os.path.join(data_config["dpath"], "qid_text_emb.pt")
        text_embedding = None
        if os.path.exists(text_emb_path):
            print(f"Loading text embeddings from {text_emb_path}")
            text_embedding = torch.load(text_emb_path, map_location=device)
        else:
            print(f"Warning: Text embedding file not found at {text_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config:
            del model_config["emb_size"]

        model = TextGMLPNoCFCausalGate(num_c=data_config["num_c"], emb_size=emb_size,
                                       text_embedding=text_embedding, emb_type=emb_type,
                                       **model_config).to(device)
    elif model_name == "text_gmlp_nocf_causal_gate_adapter":
        text_emb_path = os.path.join(data_config["dpath"], "qid_text_emb.pt")
        text_embedding = None
        if os.path.exists(text_emb_path):
            print(f"Loading text embeddings from {text_emb_path}")
            text_embedding = torch.load(text_emb_path, map_location=device)
        else:
            print(f"Warning: Text embedding file not found at {text_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config:
            del model_config["emb_size"]

        model = TextGMLPNoCFCausalGateAdapter(
            num_c=data_config["num_c"],
            emb_size=emb_size,
            text_embedding=text_embedding,
            emb_type=emb_type,
            **model_config
        ).to(device)
    elif model_name == "visual_irt":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualIRT(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_abqr":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualABQR(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_gema":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualGEMA(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_rwkv":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualRWKV(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_attn":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualAttention(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_ema":
        # Load Visual Embeddings (same logic as visual_dkt)
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        emb_size = model_config.get("emb_size", 256)
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualEMA(num_c=data_config["num_c"], emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "visual_dkt":
        # Load Visual Embeddings
        visual_emb_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        visual_embedding = None
        if os.path.exists(visual_emb_path):
            print(f"Loading visual embeddings from {visual_emb_path}")
            visual_embedding = torch.load(visual_emb_path, map_location=device)
        else:
            print(f"Warning: Visual embedding file not found at {visual_emb_path}")

        # Extract emb_size from model_config as VisualDKT expects it explicitly
        emb_size = model_config.get("emb_size", 256) # Default to 256 if not set
        if "emb_size" in model_config: del model_config["emb_size"]
            
        model = VisualDKT(num_c=data_config["num_c"], num_q=data_config.get("num_q", 0), emb_size=emb_size, 
                          visual_embedding=visual_embedding, emb_type=emb_type, 
                          **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "akt_residual":
        model = AKTResidual(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lefokt_akt":
        model = LEFOKT_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "extrakt":
        model = extraKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        LPKT, generate_qmatrix = _require_lpkt_symbols()
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "irt":
        model = IRT(num_u=_infer_num_users(data_config), num_q=data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt":
        QIKT = _require_qikt_symbol()
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "datakt":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt_residual":
        model = SimpleKTResidual(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rekt":
        model = ReKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type).to(device)
    elif model_name == "stablekt":
        model = stableKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model_config.setdefault("batch_size", 64)
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "cskt":
        model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "ukt":
        model = UKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hcgkt":
        model = HCGKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "robustkt":
        model = Robustkt(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dtransformer":
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)      
    elif model_name == "visual_akt":
        # VisualAKT needs num_q for visual features if available, otherwise num_c
        n_pid = data_config.get("num_q", 0)
        
        if "emb_type" in model_config: del model_config["emb_type"]
        if "emb_path" in model_config: del model_config["emb_path"]
        
        # 自动构建预训练特征路径，假设它在 dataset 目录下
        pretrained_path = os.path.join(data_config["dpath"], "qid_visual_emb.pt")
        
        # Pass n_pid (num_q) so we can create embeddings of correct size
        model = VisualAKT(n_question=data_config["num_c"], n_pid=n_pid, 
                          pretrained_visual_emb_path=pretrained_path,
                          **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path, strict=True, ignore_keys=None):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    ignore_keys = set(ignore_keys or [])
    if len(ignore_keys) > 0:
        net = {k: v for k, v in net.items() if k not in ignore_keys}
        missing, unexpected = model.load_state_dict(net, strict=False)
        print(f"load_model(ignore_keys={sorted(list(ignore_keys))}) -> missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(net, strict=strict)
    return model

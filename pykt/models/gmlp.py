import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, Dropout, LayerNorm

class SwiGLU(Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = Linear(in_features, hidden_features)
        self.w2 = Linear(in_features, hidden_features)
        self.w3 = Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class GMLP(Module):
    """
    GMLP v3 - The Global Context Version.
    1. Global Context: Each step perceives the global average of the sequence.
    2. Q+C Fusion: Pure and effective knowledge representation.
    3. GEMA + SwiGLU: The reliable core architecture.
    """
    def __init__(self, num_c, num_q, emb_size, dropout=0.1, emb_type='qid', **kwargs):
        super().__init__()
        self.model_name = "gmlp"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_type = emb_type
        self.emb_size = emb_size
        
        # 1. Embeddings
        self.emb_c = Embedding(num_c + 1, emb_size)
        self.concept_norm = LayerNorm(emb_size)
        if num_q > 0:
            self.emb_q = Embedding(num_q + 1, emb_size)
            self.question_norm = LayerNorm(emb_size)
        self.emb_r = Embedding(2, emb_size)
        
        # 2. Mixing Layers
        self.interaction_proj = Linear(emb_size * 2, emb_size)
        self.forget_proj = Linear(emb_size * 2, emb_size)
        self.update_proj = Linear(emb_size * 2, emb_size)
        self.fusion_gate = Linear(emb_size * 2, emb_size)
        
        # 3. Channel Mixing (SwiGLU)
        self.norm = LayerNorm(emb_size)
        self.ffn = SwiGLU(emb_size, emb_size * 2)
        self.global_alpha = nn.Parameter(torch.tensor(0.3))
        
        # 4. Prediction Layer
        self.dropout_layer = Dropout(dropout)
        self.pred_layer = nn.Sequential(
            Linear(emb_size * 2, emb_size),
            LayerNorm(emb_size),
            nn.ReLU(),
            Dropout(dropout),
            Linear(emb_size, 1)
        )

    def forward(self, q, r, qshft, c=None, cshft=None, **kwargs):
        batch_size, seq_len = q.shape
        device = q.device
        
        # --- A. Feature Extraction (Q+C Fusion) ---
        if self.num_q > 0 and c is not None:
            q_emb = self.question_norm(self.emb_q(q.long()))
            c_emb = self.concept_norm(self.emb_c(c.long()))
            q_next_emb = self.question_norm(self.emb_q(qshft.long()))
            c_next_emb = self.concept_norm(self.emb_c(cshft.long()))

            fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat([q_emb, c_emb], dim=-1)))
            fusion_gate_next = torch.sigmoid(self.fusion_gate(torch.cat([q_next_emb, c_next_emb], dim=-1)))
            v_q = c_emb + fusion_gate * q_emb
            v_q_next = c_next_emb + fusion_gate_next * q_next_emb
        else:
            v_q = self.concept_norm(self.emb_c(q.long()))
            v_q_next = self.concept_norm(self.emb_c(qshft.long()))
        v_r = self.emb_r(r.long())
        
        combined = torch.cat([v_q, v_r], dim=-1) # (B, S, 2E)
        
        # --- B. Global Context Injection ---
        # Each step now knows the 'average' vibe of the whole session
        global_avg = combined.mean(dim=1, keepdim=True) # (B, 1, 2E)
        combined = combined + self.global_alpha * global_avg
        
        # --- C. Temporal Mixing (GEMA) ---
        e = torch.relu(self.interaction_proj(combined))
        f = torch.sigmoid(self.forget_proj(combined))
        i = torch.sigmoid(self.update_proj(combined))
        
        h = torch.zeros((batch_size, self.emb_size), device=device)
        h_seq = []
        for t in range(seq_len):
            h = f[:, t, :] * h + i[:, t, :] * e[:, t, :]
            h_seq.append(h)
        h_seq = torch.stack(h_seq, dim=1)
        
        # --- D. Channel Mixing ---
        m_seq = h_seq + self.ffn(self.norm(h_seq))
        
        # --- E. Prediction ---
        predict_input = torch.cat([m_seq, v_q_next], dim=-1)
        logits = self.pred_layer(predict_input).squeeze(-1)
        y = torch.sigmoid(logits)
        
        return y

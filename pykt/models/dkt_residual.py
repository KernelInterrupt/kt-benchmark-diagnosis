import torch
from torch import nn


class DKTResidual(nn.Module):
    def __init__(
        self,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        ctw_feat_dim=64,
        ctw_delta_scale=1.0,
        symbolic_prefix="ctw",
        **kwargs,
    ):
        super().__init__()
        self.model_name = "dkt_residual"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.ctw_delta_scale = float(ctw_delta_scale)
        self.symbolic_prefix = str(symbolic_prefix).strip().lower()

        if not emb_type.startswith("qid"):
            raise ValueError("dkt_residual currently expects emb_type starting with 'qid'.")

        self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)
        self.next_concept_emb = nn.Embedding(self.num_c, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.symbolic_feature_proj = nn.Sequential(
            nn.Linear(6, ctw_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ctw_feat_dim, ctw_feat_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.emb_size + ctw_feat_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )
        self.ctw_feature_proj = self.symbolic_feature_proj

    def _shift_float_seq(self, dcur, key, default_value):
        ref = dcur["shft_cseqs"].float()
        model_device = self.interaction_emb.weight.device
        if key not in dcur:
            return torch.full_like(ref, float(default_value), device=model_device)
        return dcur[key].float().to(model_device)

    def _symbolic_key(self, suffix):
        return f"shft_{self.symbolic_prefix}_{suffix}"

    def _symbolic_features(self, dcur):
        model_device = self.interaction_emb.weight.device
        ref = dcur["shft_cseqs"].float().to(model_device)
        p_key = self._symbolic_key("pseqs")
        logit_key = self._symbolic_key("logitseqs")
        depth_key = self._symbolic_key("depthseqs")
        if depth_key not in dcur:
            depth_key = self._symbolic_key("orderseqs")
        total_key = self._symbolic_key("totalseqs")
        pos_key = self._symbolic_key("posseqs")
        neg_key = self._symbolic_key("negseqs")

        if p_key not in dcur:
            raise ValueError(
                "DKTResidual requires the CTW base-probability sequence "
                f"'{p_key}'; generate ctw_pseqs before running the residual model."
            )

        p = dcur[p_key].float().to(model_device).clamp(1e-6, 1 - 1e-6)
        if logit_key in dcur:
            logit = dcur[logit_key].float().to(model_device)
        else:
            logit = torch.logit(p)
        depth = self._shift_float_seq(dcur, depth_key, 0.0).clamp(min=0.0)
        total = torch.log1p(self._shift_float_seq(dcur, total_key, 0.0).clamp(min=0.0))
        pos = torch.log1p(self._shift_float_seq(dcur, pos_key, 0.0).clamp(min=0.0))
        neg = torch.log1p(self._shift_float_seq(dcur, neg_key, 0.0).clamp(min=0.0))
        return p, torch.stack([p, logit, depth, total, pos, neg], dim=-1)

    def _ctw_features(self, dcur):
        return self._symbolic_features(dcur)

    def forward(self, dcur, qtest=False, train=False):
        model_device = self.interaction_emb.weight.device
        c = dcur["cseqs"].long().to(model_device)
        r = dcur["rseqs"].long().to(model_device)
        cshft = dcur["shft_cseqs"].long().to(model_device)

        x = c + self.num_c * r
        xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)

        ctw_p, ctw_feat = self._symbolic_features(dcur)
        ctw_feat_proj = self.symbolic_feature_proj(ctw_feat)
        next_q_emb = self.next_concept_emb(cshft)
        delta_input = torch.cat([h, next_q_emb, ctw_feat_proj], dim=-1)
        delta = self.delta_head(delta_input).squeeze(-1)
        base_logit = torch.logit(ctw_p)
        final_logit = base_logit + self.ctw_delta_scale * delta
        preds = torch.sigmoid(final_logit)

        if train:
            return preds, 0, 0
        if qtest:
            return preds, delta_input
        return preds

import torch
from torch import nn

from .simplekt import Architecture


class SimpleKTResidual(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=8,
        separate_qa=False,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        ctw_feat_dim=64,
        ctw_delta_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "simplekt_residual"
        self.n_question = n_question
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.ctw_delta_scale = float(ctw_delta_scale)
        embed_l = d_model

        if not emb_type.startswith("qid"):
            raise ValueError("simplekt_residual currently expects emb_type starting with 'qid'.")

        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        if self.n_pid > 0 and emb_type.find("norasch") == -1:
            self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.difficult_param = None
            self.q_embed_diff = None
            self.qa_embed_diff = None

        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type="simplekt",
            seq_len=seq_len,
        )

        self.ctw_feature_proj = nn.Sequential(
            nn.Linear(6, ctw_feat_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ctw_feat_dim, ctw_feat_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(d_model + embed_l + ctw_feat_dim, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )
        self.reset()

    def reset(self):
        if self.difficult_param is not None:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def _full_seq(self, dcur, key):
        model_device = self.q_embed.weight.device
        seq = dcur[key].float().to(model_device)
        shft = dcur["shft_" + key].float().to(model_device)
        return torch.cat((seq[:, 0:1], shft), dim=1)

    def _ctw_features(self, dcur):
        required = ("ctw_pseqs", "shft_ctw_pseqs")
        missing = [key for key in required if key not in dcur]
        if missing:
            raise ValueError(
                "SimpleKTResidual requires CTW base-probability sequence columns "
                f"{', '.join(missing)}; generate ctw_pseqs before running the residual model."
            )

        p = self._full_seq(dcur, "ctw_pseqs").clamp(1e-6, 1 - 1e-6)
        if "ctw_logitseqs" in dcur:
            logit = self._full_seq(dcur, "ctw_logitseqs")
        else:
            logit = torch.logit(p)
        depth = self._full_seq(dcur, "ctw_depthseqs").clamp(min=0.0) if "ctw_depthseqs" in dcur else torch.zeros_like(p)
        total = torch.log1p(self._full_seq(dcur, "ctw_totalseqs").clamp(min=0.0)) if "ctw_totalseqs" in dcur else torch.zeros_like(p)
        pos = torch.log1p(self._full_seq(dcur, "ctw_posseqs").clamp(min=0.0)) if "ctw_posseqs" in dcur else torch.zeros_like(p)
        neg = torch.log1p(self._full_seq(dcur, "ctw_negseqs").clamp(min=0.0)) if "ctw_negseqs" in dcur else torch.zeros_like(p)
        return p, torch.stack([p, logit, depth, total, pos, neg], dim=-1)

    def forward(self, dcur, qtest=False, train=False):
        model_device = self.q_embed.weight.device
        q, c, r = dcur["qseqs"].long().to(model_device), dcur["cseqs"].long().to(model_device), dcur["rseqs"].long().to(model_device)
        qshft, cshft, rshft = dcur["shft_qseqs"].long().to(model_device), dcur["shft_cseqs"].long().to(model_device), dcur["shft_rseqs"].long().to(model_device)
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.difficult_param is not None:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(target)
            qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        d_output = self.model(q_embed_data, qa_embed_data)
        ctw_p, ctw_feat = self._ctw_features(dcur)
        ctw_feat_proj = self.ctw_feature_proj(ctw_feat)

        delta_input = torch.cat([d_output, q_embed_data, ctw_feat_proj], dim=-1)
        delta = self.delta_head(delta_input).squeeze(-1)
        base_logit = torch.logit(ctw_p)
        final_logit = base_logit + self.ctw_delta_scale * delta
        preds = torch.sigmoid(final_logit)

        if train:
            return preds, 0, 0
        if qtest:
            return preds, delta_input
        return preds

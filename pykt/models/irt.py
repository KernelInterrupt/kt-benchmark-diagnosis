import torch
from torch import nn


class IRT(nn.Module):
    def __init__(self, num_u, num_q, dropout=0.0, emb_type="qid", emb_path="", **kwargs):
        super().__init__()
        if num_u <= 0:
            raise ValueError("IRT requires num_u > 0.")
        if num_q <= 0:
            raise ValueError("IRT requires num_q > 0.")
        self.model_name = "irt"
        self.emb_type = emb_type
        self.num_u = int(num_u)
        self.num_q = int(num_q)
        self.theta = nn.Embedding(self.num_u + 1, 1)
        self.beta = nn.Embedding(self.num_q + 1, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.theta.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, dcur, qtest=False, train=False):
        if "uid" not in dcur:
            raise ValueError("IRT forward requires uid in dcur.")
        model_device = self.theta.weight.device
        q = dcur["qseqs"].long().to(model_device)
        qshft = dcur["shft_qseqs"].long().to(model_device)
        uid = dcur["uid"].long().view(-1, 1).to(model_device)

        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        theta_u = self.theta(uid).expand(-1, pid_data.size(1), -1)
        beta_q = self.beta(pid_data)
        logits = (theta_u - beta_q).squeeze(-1)
        preds = torch.sigmoid(self.dropout(logits))

        if train:
            return preds, 0, 0
        if qtest:
            hidden = torch.cat([theta_u, beta_q], dim=-1)
            return preds, hidden
        return preds

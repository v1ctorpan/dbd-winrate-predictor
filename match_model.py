import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MatchGRU(nn.Module):
    """GRU 序列分类器：变长对局序列 -> 逃生人数 5 类。"""

    def __init__(self, input_dim=30, hidden=64, num_layers=1,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, padded, lengths):
        lengths = lengths.to("cpu")
        packed = pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        idx = (lengths - 1).to(out.device).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden)
        last = out.gather(1, idx).squeeze(1)
        return self.fc(self.dropout(last))

    def predict_proba(self, padded, lengths):
        return torch.softmax(self.forward(padded, lengths), dim=1)

    def single_step(self, feat, hidden=None):
        """单步前向：feat[B,30]，可选传入 hidden，返回 (logits, hidden)。"""
        feat = feat.unsqueeze(1)  # [B,1,30]
        out, hidden = self.gru(feat, hidden)
        logits = self.fc(self.dropout(out[:, -1]))
        return logits, hidden

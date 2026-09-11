import csv
import os
import sys

import torch

import dataset_encoder as de
import match_model as mm

BASE = os.path.dirname(os.path.abspath(__file__))
TIME_NORM_HALF = 1200.0  # 与 match_dataset 一致：600s（半秒单位）


def row_features(row, t0_halfsec):
    """一行检测值 -> 30 维特征，时间每局归零后 /600s（与训练一致）。"""
    feats = de.feature_vector(row)
    feats[29] = (de.parse_time(row["frame"]) - t0_halfsec) / TIME_NORM_HALF
    return feats


class LivePredictor:
    """Stateful GRU 逐帧推理：累积 hidden，输出当前结局概率分布。"""

    def __init__(self, ckpt_path, device=None):
        self.device = device or torch.device("cpu")
        self.model = mm.MatchGRU()
        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self):
        self.hidden = None
        self.t0 = None
        self.n = 0

    def update(self, row):
        if self.t0 is None:
            self.t0 = de.parse_time(row["frame"])
        feat = torch.tensor([row_features(row, self.t0)], dtype=torch.float32,
                            device=self.device)
        with torch.no_grad():
            logits, self.hidden = self.model.single_step(feat, self.hidden)
        self.n += 1
        return torch.softmax(logits, dim=1)[0].tolist()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "models", "match_gru.pt")
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "report", "BV1Uu8z6eEVM", "detect_report.csv")
    pred = LivePredictor(ckpt)
    labels = ["0逃", "1逃", "2逃", "3逃", "4逃"]
    print("实时推理（从 CSV 逐帧喂入）。")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = pred.update(row)
            top = max(range(5), key=lambda i: p[i])
            print(f"{row['frame']}: {labels[top]} " +
                  "/".join(f"{x:.2f}" for x in p))


if __name__ == "__main__":
    main()

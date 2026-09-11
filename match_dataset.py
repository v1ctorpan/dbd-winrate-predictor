import os
import random

import torch
from torch.utils.data import Dataset

import dataset_encoder as de

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEOS = os.path.join(BASE, "dataset", "videos.jsonl")
TIME_NORM_HALF = 1200.0  # 归一化时间基准 = 600s（半秒单位）


def record_features(rec):
    """一条对局记录 -> [T, 30] 模型特征。时间分量每局归零后 /600s。"""
    frames = rec["frames"]
    feats = de.frames_to_features(frames)
    if not frames:
        return feats
    t0 = frames[0][9]
    for i, f in enumerate(frames):
        feats[i][29] = (f[9] - t0) / TIME_NORM_HALF
    return feats


class MatchDataset(Dataset):
    """加载 dataset/videos.jsonl，仅保留已标注(labe>=0)的对局。"""

    def __init__(self, videos_path=DEFAULT_VIDEOS):
        self.videos_path = videos_path
        self.records = [r for r in de.read_records(videos_path)
                        if r.get("label", -1) >= 0]
        self.keys = [f"{r['id']}:{r.get('match', 1)}" for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        feats = record_features(rec)
        return torch.tensor(feats, dtype=torch.float32), int(rec["label"])


def truncated_item(ds, i, k=None):
    """随机截断前缀采样：取 [k:T] 子序列（k 默认随机 ∈ [0, T-1]）。"""
    feats, label = ds[i]
    if k is None:
        k = random.randint(0, feats.shape[0] - 1)
    return feats[k:], label


def collate_fn(batch):
    """pad 到批内最长，返回 (padded[B,T,30], lengths[B], labels[B])。"""
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    t_max = int(lengths.max().item())
    padded = torch.zeros((len(seqs), t_max, seqs[0].shape[1]), dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    return padded, lengths, labels

# 对局序列数据管道与结局预测模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 make_report 的逐帧 CSV 检测值编码为带结局标签的 JSONL 序列数据集，训练 GRU 模型预测逃生人数（0-4），并支持实时逐帧推理。

**Architecture:** 一条数据管道 + 一个序列分类模型。`dataset_encoder.py` 把 CSV 转成 30 维特征 JSONL；`match_dataset.py` 是 PyTorch Dataset（变长批次 + 随机截断前缀采样）；`train_sequence.py` 训练 GRU；`predict_live.py` 加载权重做 stateful 逐帧推理。全部用 unittest 测试（沿用项目惯例）。

**Tech Stack:** Python 3.12、PyTorch 2.2.2、numpy、cv2（已有）、unittest（已有）。

**Spec:** `docs/superpowers/specs/2026-09-02-match-sequence-design.md`

## Global Constraints

- 特征维度固定 30 维：4 人状态 one-hot(6 类×4=24) + hooks(4) + gens(1) + 时间(1)
- 状态类别顺序：`["healthy","injured","hooked","dying","dead","escaped"]`，unknown 归入 healthy
- 时间解析：帧名 `frame_MM_SS.0.jpg` → 秒，首帧归零，归一化除以 600
- gens 列值可能是字符串 `"None"` → 特征填 -1
- 结局标签 = 逃生人数 0-4（多分类 5 类）
- 数据切分**按局**，不按帧
- 测试运行命令：`/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest discover -s tests`
- 种子数据：`report/BV1Uu8z6eEVM/detect_report.csv`（label=3）和 `report/BV16QtT6ZEPq/detect_report.csv`（label=1）

---

### Task 1: 数据集编码器 dataset_encoder.py

**Files:**
- Create: `dataset_encoder.py`
- Test: `tests/test_dataset_encoder.py`

**Interfaces:**
- Consumes: `report/{match}/detect_report.csv`（列：frame,scale,p1,p2,p3,p4,hooks,gens,机器标注）
- Produces: `dataset/matches/{match_id}.jsonl`（每行一帧 `{"t":int,"features":[30 floats]}`，末行 `{"label":int,"n_frames":int}`）
- 函数 `encode_csv_to_jsonl(csv_path, out_path, label)` 和 `parse_time(fname)`、`one_hot_state(state)`、`feature_vector(row)`（供测试直接调用）

- [ ] **Step 1: 写失败测试**

```python
import csv
import json
import os
import tempfile
import unittest

import dataset_encoder as de


class TestParseTime(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(de.parse_time("frame_00_00.0.jpg"), 0)
        self.assertEqual(de.parse_time("frame_01_30.0.jpg"), 90)
        self.assertEqual(de.parse_time("frame_12_45.0.jpg"), 765)


class TestOneHotState(unittest.TestCase):
    def test_one_hot(self):
        v = de.one_hot_state("injured")
        self.assertEqual(len(v), 6)
        self.assertEqual(v, [0, 1, 0, 0, 0, 0])

    def test_unknown_maps_to_healthy(self):
        v = de.one_hot_state("unknown")
        self.assertEqual(v[0], 1)


class TestFeatureVector(unittest.TestCase):
    def test_dimension_30(self):
        row = {"p1": "healthy", "p2": "injured", "p3": "hooked",
               "p4": "dying", "hooks": "1/2/0/1", "gens": "3", "frame": "frame_01_00.0.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(len(v), 30)

    def test_gens_none_to_minus1(self):
        row = {"p1": "healthy", "p2": "healthy", "p3": "healthy",
               "p4": "healthy", "hooks": "0/0/0/0", "gens": "None", "frame": "frame_00_00.0.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(v[28], -1.0)


class TestEncodeCsv(unittest.TestCase):
    def _write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "scale", "p1", "p2", "p3", "p4", "hooks", "gens", "机器标注"])
            w.writerow(["frame_00_00.0.jpg", "1.0", "healthy", "healthy", "healthy", "healthy", "0/0/0/0", "5", "正常"])
            w.writerow(["frame_00_10.0.jpg", "1.0", "injured", "healthy", "healthy", "healthy", "0/0/0/0", "5", "正常"])

    def test_encode_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "in.csv")
            out_path = os.path.join(d, "out.jsonl")
            self._write_csv(csv_path)
            de.encode_csv_to_jsonl(csv_path, out_path, label=4)
            lines = [json.loads(l) for l in open(out_path, encoding="utf-8")]
            self.assertEqual(lines[0]["t"], 0)
            self.assertEqual(lines[0]["features"][28], 5.0)
            self.assertEqual(lines[1]["t"], 10)
            self.assertEqual(lines[-1], {"label": 4, "n_frames": 2})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_dataset_encoder -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'dataset_encoder'`）

- [ ] **Step 3: 实现 dataset_encoder.py**

```python
import csv
import json
import os

STATES = ["healthy", "injured", "hooked", "dying", "dead", "escaped"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
TIME_NORM = 600.0


def parse_time(fname):
    base = fname.split(".")[0]          # frame_MM_SS
    parts = base.split("_")
    mm, ss = int(parts[1]), int(parts[2])
    return mm * 60 + ss


def one_hot_state(state):
    idx = STATE_TO_IDX.get(state, 0)    # unknown -> healthy(0)
    v = [0.0] * len(STATES)
    v[idx] = 1.0
    return v


def feature_vector(row):
    feats = []
    for p in ("p1", "p2", "p3", "p4"):
        feats.extend(one_hot_state(row[p]))
    for h in row["hooks"].split("/"):
        feats.append(float(h))
    gens = row["gens"].strip()
    feats.append(-1.0 if gens in ("None", "") else float(gens))
    feats.append(parse_time(row["frame"]) / TIME_NORM)
    return feats


def encode_csv_to_jsonl(csv_path, out_path, label):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {"t": parse_time(r["frame"]), "features": feature_vector(r)}
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"label": int(label), "n_frames": len(rows)}) + "\n")
    return len(rows)


def main():
    spec = {
        "BV1Uu8z6eEVM": ("report/BV1Uu8z6eEVM/detect_report.csv", 3),
        "BV16QtT6ZEPq": ("report/BV16QtT6ZEPq/detect_report.csv", 1),
    }
    for match_id, (csv_path, label) in spec.items():
        out = os.path.join("dataset", "matches", f"{match_id}.jsonl")
        n = encode_csv_to_jsonl(csv_path, out, label)
        print(f"[{match_id}] {n} frames -> {out} label={label}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_dataset_encoder -v`
Expected: PASS（4 tests）

- [ ] **Step 5: 生成真实种子数据并验证**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python dataset_encoder.py`
Expected: 输出两局 JSONL，打印帧数（BV1=110, BV16=30）

- [ ] **Step 6: 提交**

```bash
git add dataset_encoder.py tests/test_dataset_encoder.py dataset/matches/
git commit -m "feat: CSV到JSONL数据集编码器(30维特征+结局标签)"
```

---

### Task 2: PyTorch 数据集加载器 match_dataset.py

**Files:**
- Create: `match_dataset.py`
- Test: `tests/test_match_dataset.py`

**Interfaces:**
- Consumes: `dataset/matches/{match_id}.jsonl`（Task 1 产出）
- Produces:
  - `MatchDataset(match_dir)` → `len()`、`__getitem__(i)` 返回 `(features_tensor[T,30], label_int)`
  - `collate_fn(batch)` → `(padded[B,T,30], lengths[B], labels[B])`（pad 到批内最长 + mask）
  - `truncated_item(dataset, i, k)` → 随机截断前缀采样：取 `features[k:]` 作为子序列（k 随机 ∈ [0,T-1]）

- [ ] **Step 1: 写失败测试**

```python
import json
import os
import tempfile
import unittest

import torch

import match_dataset as md


def _write_sample(match_dir, match_id, n_frames, label):
    path = os.path.join(match_dir, f"{match_id}.jsonl")
    os.makedirs(match_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_frames):
            rec = {"t": i * 10, "features": [float(j) for j in range(30)]}
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"label": label, "n_frames": n_frames}) + "\n")


class TestMatchDataset(unittest.TestCase):
    def test_len_and_getitem(self):
        with tempfile.TemporaryDirectory() as d:
            _write_sample(d, "m1", 5, 2)
            _write_sample(d, "m2", 3, 4)
            ds = md.MatchDataset(d)
            self.assertEqual(len(ds), 2)
            feats, label = ds[0]
            self.assertEqual(feats.shape, (5, 30))
            self.assertEqual(label, 2)

    def test_collate_padding(self):
        with tempfile.TemporaryDirectory() as d:
            _write_sample(d, "m1", 5, 2)
            _write_sample(d, "m2", 3, 4)
            ds = md.MatchDataset(d)
            batch = md.collate_fn([ds[0], ds[1]])
            padded, lengths, labels = batch
            self.assertEqual(padded.shape, (2, 5, 30))
            self.assertEqual(lengths.tolist(), [5, 3])
            self.assertEqual(labels.tolist(), [2, 4])

    def test_truncated_item(self):
        with tempfile.TemporaryDirectory() as d:
            _write_sample(d, "m1", 5, 2)
            ds = md.MatchDataset(d)
            feats, label = md.truncated_item(ds, 0, k=2)
            self.assertEqual(feats.shape, (3, 30))
            self.assertEqual(feats[0, 0], 2.0)   # 原第2帧（从0起）
            self.assertEqual(label, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_match_dataset -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 match_dataset.py**

```python
import json
import os
import random

import torch
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    def __init__(self, match_dir):
        self.match_dir = match_dir
        self.items = []
        for fn in sorted(os.listdir(match_dir)):
            if fn.endswith(".jsonl"):
                self.items.append(os.path.join(match_dir, fn))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        feats, label = self._load(self.items[i])
        return torch.tensor(feats, dtype=torch.float32), label

    def _load(self, path):
        feats, label = [], None
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "label" in rec:
                    label = rec["label"]
                else:
                    feats.append(rec["features"])
        return feats, label


def truncated_item(ds, i, k=None):
    """随机截断前缀采样：取 [k:T] 子序列（k 默认随机 ∈ [0, T-1]）。"""
    feats, label = ds[i]
    if k is None:
        k = random.randint(0, feats.shape[0] - 1)
    return feats[k:], label


def collate_fn(batch):
    """pad 到批内最长，返回 (padded, lengths, labels)。"""
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    t_max = lengths.max().item()
    padded = torch.zeros((len(seqs), t_max, seqs[0].shape[1]), dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    return padded, lengths, labels
```

- [ ] **Step 4: 运行确认通过**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_match_dataset -v`
Expected: PASS（3 tests）

- [ ] **Step 5: 提交**

```bash
git add match_dataset.py tests/test_match_dataset.py
git commit -m "feat: 变长序列PyTorch数据集(截断前缀采样+pad/collate)"
```

---

### Task 3: GRU 序列分类模型 match_model.py

**Files:**
- Create: `match_model.py`
- Test: `tests/test_match_model.py`

**Interfaces:**
- Consumes: `match_dataset.py` 的 `collate_fn` 输出 `(padded[B,T,30], lengths[B], labels[B])`
- Produces:
  - `MatchGRU(input_dim=30, hidden=64, num_layers=1, num_classes=5, dropout=0.3)`
  - `forward(padded, lengths)` → `logits[B,5]`（用 `pack_padded_sequence` 处理变长）
  - `predict_proba(model, padded, lengths)` → `proba[B,5]`（softmax）
  - 方便推理的单步接口 `single_step(model, feat[B,30], hidden=None)` → `(logits, hidden)`（Task 5 用）

- [ ] **Step 1: 写失败测试**

```python
import unittest

import torch

import match_model as mm


class TestMatchGRU(unittest.TestCase):
    def test_forward_shape(self):
        model = mm.MatchGRU()
        padded = torch.randn(2, 5, 30)
        lengths = torch.tensor([5, 3])
        logits = model.forward(padded, lengths)
        self.assertEqual(logits.shape, (2, 5))

    def test_predict_proba(self):
        model = mm.MatchGRU()
        padded = torch.randn(2, 5, 30)
        lengths = torch.tensor([5, 3])
        proba = model.predict_proba(padded, lengths)
        self.assertEqual(proba.shape, (2, 5))
        self.assertTrue(torch.allclose(proba.sum(dim=1), torch.ones(2), atol=1e-4))

    def test_single_step(self):
        model = mm.MatchGRU()
        feat = torch.randn(1, 30)
        logits, hidden = model.single_step(feat)
        self.assertEqual(logits.shape, (1, 5))
        self.assertEqual(hidden.shape, (1, 1, 64))
        logits2, _ = model.single_step(feat, hidden)
        self.assertEqual(logits2.shape, (1, 5))

    def test_variable_length_uses_full_sequence(self):
        # 长序列的预测应不同于只用第一帧的预测（证明不是只看首帧）
        model = mm.MatchGRU()
        padded = torch.randn(1, 8, 30)
        lengths = torch.tensor([8])
        full_logits = model.forward(padded, lengths)
        single_logits = model.forward(padded[:, :1], torch.tensor([1]))
        self.assertFalse(torch.allclose(full_logits, single_logits, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_match_model -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 match_model.py**

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MatchGRU(nn.Module):
    def __init__(self, input_dim=30, hidden=64, num_layers=1,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden, num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, padded, lengths):
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
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
```

- [ ] **Step 4: 运行确认通过**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_match_model -v`
Expected: PASS（4 tests）

- [ ] **Step 5: 提交**

```bash
git add match_model.py tests/test_match_model.py
git commit -m "feat: GRU序列分类模型(变长pack_padded+单步推理接口)"
```

---

### Task 4: 训练脚本 train_sequence.py

**Files:**
- Create: `train_sequence.py`
- Test: `tests/test_train_sequence.py`

**Interfaces:**
- Consumes: `match_dataset.MatchDataset` + `collate_fn` + `truncated_item`、`match_model.MatchGRU`
- Produces:
  - `train_one_epoch(model, dataset, opt, criterion, batch_size, device)` → 平均 loss
  - `evaluate(model, dataset, device)` → (accuracy, avg_loss)
  - `train(match_dir, out_ckpt, epochs, seed)` → 保存 `model.state_dict()` 到 out_ckpt，返回 val 指标
  - 脚本 main：加载 `dataset/matches/`，按局切分 train/val（`random.sample` 种子固定），训练后保存 `models/match_gru.pt`

- [ ] **Step 1: 写失败测试**

```python
import json
import os
import tempfile
import unittest

import torch

import train_sequence as ts


def _write_match(match_dir, match_id, n_frames, label):
    path = os.path.join(match_dir, f"{match_id}.jsonl")
    os.makedirs(match_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_frames):
            rec = {"t": i * 10, "features": [1.0] * 30}
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"label": label, "n_frames": n_frames}) + "\n")


class TestTrain(unittest.TestCase):
    def test_split_by_match(self):
        with tempfile.TemporaryDirectory() as d:
            for mid, n, lbl in [("a", 4, 0), ("b", 4, 1), ("c", 4, 2), ("d", 4, 3), ("e", 4, 4)]:
                _write_match(d, mid, n, lbl)
            train_ids, val_ids = ts.split_matches(d, val_frac=0.2, seed=42)
            self.assertEqual(len(train_ids), 4)
            self.assertEqual(len(val_ids), 1)
            self.assertTrue(set(train_ids).isdisjoint(set(val_ids)))

    def test_train_loop_runs(self):
        with tempfile.TemporaryDirectory() as d:
            for mid, n, lbl in [("a", 4, 0), ("b", 4, 1)]:
                _write_match(d, mid, n, lbl)
            train_ids, _ = ts.split_matches(d, val_frac=0.0, seed=42)
            model, opt, criterion = ts.make_training(torch.device("cpu"))
            loss = ts.train_one_epoch(model, d, train_ids, opt, criterion,
                                      batch_size=2, device=torch.device("cpu"))
            self.assertGreater(loss, 0.0)

    def test_save_load_ckpt(self):
        model = ts.make_model()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            torch.save(model.state_dict(), ckpt)
            model2 = ts.make_model()
            model2.load_state_dict(torch.load(ckpt, weights_only=True))
            self.assertTrue(os.path.exists(ckpt))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_train_sequence -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 train_sequence.py**

```python
import argparse
import os
import random

import torch
import torch.nn as nn

import match_dataset as md
import match_model as mm


def split_matches(match_dir, val_frac=0.1, seed=42):
    # 按局随机切分（不按帧）。数据达千局且各类均衡后可升级为按 label 分层抽样。
    ids = sorted(f[:-6] for f in os.listdir(match_dir) if f.endswith(".jsonl"))
    rng = random.Random(seed)
    n_val = max(1, int(len(ids) * val_frac)) if val_frac > 0 else 0
    val_ids = set(rng.sample(ids, n_val))
    train_ids = [i for i in ids if i not in val_ids]
    return train_ids, sorted(val_ids)


def make_model():
    return mm.MatchGRU()


def make_training(device):
    model = mm.MatchGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    return model, opt, crit


def train_one_epoch(model, match_dir, train_ids, opt, criterion,
                    batch_size=4, device=torch.device("cpu")):
    model.train()
    dataset = md.MatchDataset(match_dir)
    id_to_idx = {os.path.splitext(os.path.basename(p))[0]: i
                 for i, p in enumerate(dataset.items)}
    idxs = [id_to_idx[i] for i in train_ids]
    random.shuffle(idxs)
    total = 0.0
    count = 0
    for b in range(0, len(idxs), batch_size):
        batch_ids = idxs[b:b + batch_size]
        sub = [md.truncated_item(dataset, i) for i in batch_ids]
        padded, lengths, labels = md.collate_fn(sub)
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        opt.zero_grad()
        logits = model.forward(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def evaluate(model, match_dir, ids, device=torch.device("cpu")):
    model.eval()
    dataset = md.MatchDataset(match_dir)
    id_to_idx = {os.path.splitext(os.path.basename(p))[0]: i
                 for i, p in enumerate(dataset.items)}
    idxs = [id_to_idx[i] for i in ids]
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i in idxs:
            feats, label = md.truncated_item(dataset, i, k=0)  # 完整序列评估
            padded, lengths, labels = md.collate_fn([(feats, label)])
            logits = model.forward(padded.to(device), lengths.to(device))
            loss_sum += nn.CrossEntropyLoss()(logits, labels.to(device)).item()
            pred = logits.argmax(dim=1).item()
            correct += (pred == label)
            total += 1
    return correct / max(total, 1), loss_sum / max(total, 1)


def train(match_dir="dataset/matches", out_ckpt="models/match_gru.pt",
          epochs=30, seed=42, device=None):
    if device is None:
        device = torch.device("cpu")
    train_ids, val_ids = split_matches(match_dir, val_frac=0.1, seed=seed)
    model, opt, criterion = make_training(device)
    best_acc = 0.0
    for ep in range(epochs):
        tr_loss = train_one_epoch(model, match_dir, train_ids, opt, criterion,
                                  device=device)
        val_acc, val_loss = evaluate(model, match_dir, val_ids, device=device)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
        print(f"epoch {ep}: train_loss={tr_loss:.4f} val_acc={val_acc:.3f} val_loss={val_loss:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", default="dataset/matches")
    parser.add_argument("--out", default="models/match_gru.pt")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.match_dir, args.out, args.epochs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_train_sequence -v`
Expected: PASS（3 tests）

- [ ] **Step 5: 用种子数据真实训练一次**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python train_sequence.py --epochs 30`
Expected: 训练循环打印各 epoch 指标，最终保存 `models/match_gru.pt`

- [ ] **Step 6: 提交**

```bash
git add train_sequence.py tests/test_train_sequence.py models/match_gru.pt
git commit -m "feat: GRU训练脚本(按局切分+截断前缀训练+checkpoint)"
```

---

### Task 5: 实时推理 predict_live.py

**Files:**
- Create: `predict_live.py`
- Test: `tests/test_predict_live.py`

**Interfaces:**
- Consumes: `match_model.MatchGRU` + `single_step`、`dataset_encoder.feature_vector`、检查点 `models/match_gru.pt`
- Produces:
  - `LivePredictor(ckpt_path)`：封装 stateful GRU
  - `LivePredictor.update(row)` → `proba[5]`（喂入一行检测值，累积 hidden，返回当前预测分布）
  - `LivePredictor.reset()` → 清空 hidden 和内部累计特征（新局开始用）

- [ ] **Step 1: 写失败测试**

```python
import json
import os
import tempfile
import unittest

import torch

import predict_live as pl


def _make_ckpt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from match_model import MatchGRU
    torch.save(MatchGRU().state_dict(), path)


class TestLivePredictor(unittest.TestCase):
    def _row(self):
        return {"frame": "frame_01_00.0.jpg", "p1": "healthy", "p2": "injured",
                "p3": "hooked", "p4": "dying", "hooks": "1/2/0/1", "gens": "3"}

    def test_update_returns_proba(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            p = pred.update(self._row())
            self.assertEqual(len(p), 5)
            self.assertAlmostEqual(sum(p), 1.0, places=4)

    def test_stateful_accumulation(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            r1 = pred.update(self._row())
            r2 = pred.update(self._row())
            self.assertEqual(len(r1), 5)
            self.assertEqual(len(r2), 5)

    def test_reset(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            pred.update(self._row())
            pred.reset()
            p = pred.update(self._row())
            self.assertEqual(len(p), 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_predict_live -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 predict_live.py**

```python
import json
import os

import torch

import dataset_encoder as de
import match_model as mm


class LivePredictor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or torch.device("cpu")
        self.model = mm.MatchGRU()
        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self):
        self.hidden = None
        self.n = 0

    def update(self, row):
        feat = torch.tensor([de.feature_vector(row)], dtype=torch.float32,
                            device=self.device)
        with torch.no_grad():
            logits, self.hidden = self.model.single_step(feat, self.hidden)
        self.n += 1
        return torch.softmax(logits, dim=1)[0].tolist()


def main():
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/match_gru.pt"
    pred = LivePredictor(ckpt)
    labels = ["0逃", "1逃", "2逃", "3逃", "4逃"]
    print("实时推理（从 CSV 逐帧喂入）。Ctrl-C 退出。")
    import csv
    path = sys.argv[2] if len(sys.argv) > 2 else "report/BV1Uu8z6eEVM/detect_report.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = pred.update(row)
            top = max(range(5), key=lambda i: p[i])
            print(f"{row['frame']}: {labels[top]} {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}/{p[3]:.2f}/{p[4]:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest tests.test_predict_live -v`
Expected: PASS（3 tests）

- [ ] **Step 5: 用真实检查点冒烟测试推理**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python predict_live.py models/match_gru.pt report/BV1Uu8z6eEVM/detect_report.csv`
Expected: 逐帧打印预测分布，随帧数增加预测演化

- [ ] **Step 6: 全量测试回归**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -m unittest discover -s tests`
Expected: 25 旧测试 + 本项目 ~17 新测试全部 PASS

- [ ] **Step 7: 提交**

```bash
git add predict_live.py tests/test_predict_live.py
git commit -m "feat: 实时stateful推理预测器(逐帧输出结局分布)"
```

---

### Task 6: 环境修复 + 文档收尾

**Files:**
- Modify: `PROGRESS.md`（更新 gens→序列管道进展）
- Modify: `docs/superpowers/specs/2026-09-02-match-sequence-design.md`（如需）
- Create: `requirements.txt`（`torch==2.2.2`，注明需修复安装）

**Interfaces:**
- Consumes: 前 5 个任务的产出

- [ ] **Step 1: 修复 torch 安装**

问题：`torch` 已安装但 `libtorch_cpu.dylib` 缺失（pip 安装损坏）。

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -c "import torch; print(torch.__version__)"`
若仍报 ImportError，修复：`/Users/wypan/opt/anaconda3/envs/dbd/bin/pip install --force-reinstall torch==2.2.2`
Expected: 打印 torch 版本，无 ImportError

- [ ] **Step 2: 验证 import 正常**

Run: `/Users/wypan/opt/anaconda3/envs/dbd/bin/python -c "import torch, torch.nn; print('ok')"`
Expected: `ok`

- [ ] **Step 3: 写 requirements.txt**

```
torch==2.2.2
numpy==1.26.0
opencv-python==4.8.0
```

- [ ] **Step 4: 更新 PROGRESS.md**

在已完成表格加一行：`序列数据管道 | dataset_encoder.py / match_dataset.py / train_sequence.py / predict_live.py | ✅ 编码器/数据集/GRU训练/实时推理跑通，种子数据 BV1+BV16 两局`
并更新文件头日期/状态。

- [ ] **Step 5: 提交**

```bash
git add requirements.txt PROGRESS.md
git commit -m "docs: 序列管道进展记录+依赖清单"
```

---

## 执行后验收

1. `dataset/matches/BV1Uu8z6eEVM.jsonl` 存在，首帧 features 长度=30，末行 label=3
2. `dataset/matches/BV16QtT6ZEPq.jsonl` 存在，末行 label=1
3. `models/match_gru.pt` 存在（训练产出）
4. `predict_live.py` 逐帧输出 5 类概率分布
5. 全量 unittest PASS
6. 全部任务已 commit

# 对局序列数据管道与结局预测模型 — 设计文档

> 日期：2026-09-02
> 状态：已获用户批准（brainstorming 流程）

## 1. 背景与目标

现有 `make_report.py` 已将每局视频抽帧识别为结构化检测值（玩家状态×4、hooks×4、gens、时间），
输出逐帧 CSV。下一步：把这些检测值累积成**带结局标签的对局序列数据集**，训练模型**预测结算结局**。

**预测目标**：结算结局 = **逃生人数 0-4**（5 类多分类）。

**训练/推理模式**（用户确认）：
- 训练：输入完整对局序列 → 输出结局
- 推理：实时逐帧，用"开局到当前"的前缀序列预测结局（每帧可刷新预测）

**数据规模**（用户确认）：最终上千局带结局标注。

## 2. 核心设计决策

1. **一局 = 一个变长序列样本**（T×F 矩阵 + 1 个标签），不定长序列是系统的一等公民。
2. **存储格式 = JSONL**：每局一个 `.jsonl` 文件，每行一帧，末尾一行标签。
3. **特征 = 30 维定长向量**：4 人状态 one-hot(6类×4) + hooks 计数(4) + gens(1) + 归一化时间(1)。
4. **模型 = GRU 序列分类器**：天然支持不定长、stateful 逐帧推理、千局数据量友好。
5. **训练分布对齐**：随机截断前缀采样，让模型学会"任意进度预测结局"，与实时推理一致。
6. **技术栈**：新增 PyTorch（当前仅 cv2/numpy/matplotlib/PIL）。

## 3. 数据格式

### 3.1 目录布局

```
dataset/
  matches/
    BV1Uu8z6eEVM.jsonl      # 每局一个文件
    BV16QtT6ZEPq.jsonl
    ...
  index.csv                 # 可选：match_id, label, n_frames 摘要
```

### 3.2 单局 JSONL 结构

```
# 每行一帧（features = 30 维定长向量）
{"frame": "frame_00_00", "t": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0, "hooks": [0,0,0,0], "gens": 5}
{"frame": "frame_00_10", "t": 10, ...}
...
# 末行（标签）
{"label": 3, "n_frames": 110}
```

- `t` = 局内秒数（从文件名 `frame_MM_SS` 解析，首帧归零）
- 标签 `label` = 逃生人数 0-4（从结算画面自动标注，见 3.4）
- JSONL 相比 CSV 的优势：每局独立、可增量追加、天然含结构（嵌套 hooks 数组、标签混排）

### 3.3 特征编码（30 维，模型输入）

| 分量 | 维度 | 编码 | 说明 |
|---|---|---|---|
| p1~p4 状态 | 6×4=24 | one-hot | healthy/injured/hooked/dying/dead/escaped（unknown 归入 healthy） |
| hooks | 4 | 数值 0/1/2 | 每人上钩次数 |
| gens | 1 | 数值 0-5 | 剩余发电机 |
| 时间 t | 1 | 归一化 | t / T_typical（如 /600s） |

- one-hot 由数据集编码器从原始检测值生成（`make_report` 输出 → 30 维特征）
- 归一化时间使模型能区分"开局 vs 末局同状态"的语义差异
- one-hot 与时间归一化都在**编码阶段**完成；gens/hooks 为自然计数保持原样，训练脚本内可再作标准化（可选）

### 3.4 结局标注

- 从**结算画面**自动标注（后续单独实现，见项目待办）：检测结算界面 → 统计逃生人数 0-4
- 种子数据：BV1/BV16 两局结局由人工标注（test1 为 720p 随机采样帧，非完整对局，不作为序列样本）
- 标签写入 JSONL 末行，索引表记录 label 来源（auto/manual）

## 4. 模型架构

### 4.1 GRU 序列分类器

```
输入 (B, T, F=30)
  → GRU(input=30, hidden=64, num_layers=1~2, batch_first=True)
  → 取最后时间步 hidden[-1]                     # 变长用 pack_padded_sequence
  → Dropout(p=0.3)
  → Linear(64 → 5)
  → Softmax → 5 类（逃生人数 0-4）
```

- 不定长处理：批次内 `pad` 到最长 + `pack_padded_sequence` 让 GRU 忽略 pad
- hidden 层数先 1 层（千局数据、30 维小特征，过深易过拟合）

### 4.2 训练方式

- 损失：CrossEntropyLoss（序数可后加 MAE 评估，不改变损失）
- 优化：Adam，lr=1e-3，batch 按序列数（如 32）
- **随机截断前缀采样**（关键）：
  - 每个 epoch 对每条完整序列随机取起点 k ∈ [0, T]，用 `[k:T]` 子序列训练
  - 模型学到"任意进度 → 结局预测"，与实时推理前缀分布一致
  - 避免训练全序列 / 推理前缀的分布偏移
- Epoch 数：early stopping on val

### 4.3 实时推理（Stateful GRU）

- 训练完成后导出权重
- 推理时：初始化 hidden=0 → 每帧 feed 该帧 30 维特征 → 单步 forward → 输出当前预测分布
- 每帧 O(hidden) 计算，毫秒级，适合实时间隔截图
- 训练（截断重算）与推理（stateful 累积）是标准模式分离，不冲突
- 首帧预测即给出 baseline；随序列增长预测逐渐收敛

### 4.4 数据切分

- **按局切分**（不按帧）：避免同局帧泄漏 train/val
- 上千局：80/10/10 train/val/test（stratify by label）
- 指标：Accuracy + 混淆矩阵 + MAE（0-4 序数误差）

## 5. 技术栈与新增文件

新增依赖：`torch`（PyTorch）。

| 文件 | 用途 |
|---|---|
| `dataset_encoder.py` | make_report CSV → JSONL 特征序列 + 标签 |
| `match_dataset.py` | PyTorch Dataset：加载 JSONL、变长批次、截断前缀采样、pad+mask |
| `train_sequence.py` | 训练/验证/评估循环 |
| `predict_live.py` | 实时推理：stateful GRU，逐帧输出结局预测 |
| `dataset/matches/` | 生成的序列数据 |
| 模型权重 | `models/match_gr.pt` |

## 6. 验收标准

1. `dataset_encoder.py` 能把现有 BV1/BV16 报告转成 JSONL，特征维度 = 30，与设计一致
2. 数据集加载器正确支持变长批次（padding + pack_padded_sequence）
3. 训练脚本可在种子数据（≥3 局）上跑通全流程，输出模型权重
4. 实时推理脚本逐帧输出 5 类分布，且随帧数增加预测合理演化
5. 模型评估按局切分，报告 accuracy/混淆矩阵/MAE

## 7. 后续迭代（不在本期范围）

- 结算画面自动标注器（当前人工标注种子数据）
- 数据扩充：更多视频抽帧 → 编码 → 标注
- 模型超参调优（hidden 层数、dropout、lr）
- 前端：胜率走势图渲染

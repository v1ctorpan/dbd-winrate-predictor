# 数据集格式说明 — dataset/videos.jsonl

> 日期：2026-09-02
> 单文件存储，每行一条视频（一局），供序列模型（GRU）训练与实时推理。

## 1. 目录布局

```
dataset/
  videos.jsonl      # 全部已标注视频，每行一条
```

## 2. 行结构

每行一个 JSON 对象，表示一条视频（一整局对局）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 视频编号，如 `BV1Uu8z6eEVM` |
| `title` | string | 视频标题，可截断。暂为空，后续从 bilibili 抓取填充 |
| `url` | string | 视频链接，如 `https://www.bilibili.com/video/BV1Uu8z6eEVM`。暂为空，后续填充 |
| `features` | float[][30] | 逐帧 30 维特征，`features[i]` = 第 i 帧，长度 = 该局帧数 T |
| `label` | int | 结局标签 = 逃生人数 0–4（5 类多分类目标） |

示例：

```json
{"id": "BV1Uu8z6eEVM", "title": "", "url": "",
 "features": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, "…共30维…"],
              ["…第2帧…"], "…共110帧…"],
 "label": 3}
```

## 3. 特征向量（30 维定长）

| 分量 | 下标 | 维度 | 编码 | 说明 |
|---|---|---|---|---|
| p1~p4 状态 | 0–23 | 6×4=24 | one-hot | 每玩家 6 维；类别顺序 `healthy/injured/hooked/dying/dead/escaped`；unknown 归入 healthy |
| hooks | 24–27 | 4 | 数值 0/1/2 | p1~p4 上钩次数 |
| gens | 28 | 1 | 数值 0–5 | 剩余发电机数；无 HUD 时填 `-1.0` |
| 时间 | 29 | 1 | 原始秒数 | 局内秒数 t，首帧 0，不归一化 |

- 由 `dataset_encoder.py` 从 `make_report.py` 的逐帧 CSV 编码生成
- one-hot 在编码阶段完成；hooks/gens/时间为自然计数/秒数保持原样

## 4. 时间语义

- 时间分量已内嵌在 `features[29]`：从帧名解析 `frame_MM_SS.0.jpg` → `MM*60 + SS` 秒，首帧 0
- 抽样间隔：当前约 10s/帧（`extract_frames.py`）

## 5. 结局标注（label）

- `label` = 结算画面逃生人数 0–4
- 种子数据人工标注：BV1 → 3，BV16 → 1
- 后续：从结算画面自动标注；`title`/`url` 从 bilibili 抓取

## 6. 生成方式

```bash
# 需先有 report/{视频ID}/detect_report.csv（make_report.py 产出）
python dataset_encoder.py
# 输出 dataset/videos.jsonl
```

新增视频流程：抽帧 → make_report 生成 CSV → 人工标注 label → 在 `dataset_encoder.py` main 的 spec 追加一行 → 重跑。

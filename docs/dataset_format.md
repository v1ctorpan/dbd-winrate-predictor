# 数据集格式说明 — dataset/videos.jsonl

> 日期：2026-09-02（2026-09-06 更新：新增 `match` 字段；`title`/`url` 由产线自动填充）
> 单文件存储，每行一条视频（一局），供序列模型（GRU）训练与实时推理。

## 1. 目录布局

```
dataset/
  videos.jsonl      # 全部已标注视频，每行一条
```

## 2. 行结构

每行一个 JSON 对象，表示一场对局样本。字段顺序固定：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 视频 BV 号，如 `BV1Uu8z6eEVM` |
| `title` | string | 视频标题。产线运行时自动从 `picture/raw_videos/{id}.info.json` 填充；无则空串 |
| `url` | string | canonical 视频链接 `https://www.bilibili.com/video/{id}`。产线运行时自动填充；无则空串 |
| `match` | int | 局号。视频内含多局时 1..N；整段式视频（一视频一局）固定 1 |
| `features` | float[][30] | 逐帧 30 维特征，`features[i]` = 第 i 帧，长度 = 该局帧数 T |
| `label` | int | 结局标签 = 逃生人数 0–4（5 类多分类目标）；产线自动分段未标注局为 -1 |

示例：

```json
{"id": "BV1Uu8z6eEVM", "title": "", "url": "", "match": 1,
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
- 后续：从结算画面自动标注；`title`/`url` 在产线运行时自动填充（见下）

## 6. 生成方式

- 数据集手工/种子记录由 `dataset_encoder.py` 的 `encode_csv` 生成（支持传 `meta` 覆盖 `match/title/url`）
- 产线自动写入：`run_pipeline.py` 在把自动分段写入 jsonl 前，通过 `_resolve_meta` 读取
  `picture/raw_videos/{bvid}.info.json`（`title`、`webpage_url`）填充 `title`/`url`，并把局号写入 `match`；
  `--title`/`--url` 可显式覆盖，缺省 url 兜底为 `https://www.bilibili.com/video/{bvid}`

```bash
python dataset_encoder.py
# 输出 dataset/videos.jsonl（手工种子记录）
python run_pipeline.py --video ... # 产线自动分段 + 写入（title/url/match 自动填充）
```

新增视频流程：下载（含 info.json）→ 抽帧 → make_report 生成 CSV → 人工标注 label → 追加写入。

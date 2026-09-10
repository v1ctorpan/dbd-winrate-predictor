# 数据集格式说明 — dataset/videos.jsonl

> 日期：2026-09-02（2026-09-08 更新：`features` 30 维改为紧凑 `frames` 帧元组，体积约缩 4.5x）
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
| `frames` | int[][10] | 逐帧**紧凑元组**（见 §3），`frames[i]` = 第 i 帧，长度 = 该局帧数 T |
| `label` | int | 结局标签 = 逃生人数 0–4（5 类多分类目标）；**由 `labeling.infer_label_from_csv` 从该局末尾 HUD 帧自动推断**（结尾若干帧中出现过 `escaped` 的人数）；无法判定（如末尾无有效 HUD、剪辑向片段）为 -1 |

示例：

```json
{"id": "BV1Uu8z6eEVM", "title": "", "url": "", "match": 1,
 "frames": [[0,0,0,0, 0,0,0,0, 5, 0], [0,1,0,0, 0,0,0,0, 5, 20]],
 "label": 2}
```

## 3. 帧元组（紧凑，10 个整数）

`frames[i] = [s1, s2, s3, s4, h1, h2, h3, h4, gens, t_half]`

| 下标 | 字段 | 取值 | 说明 |
|---|---|---|---|
| 0–3 | p1~p4 状态 | 类别索引 0–5 | `healthy/injured/hooked/dying/dead/escaped` 的顺序索引；`unknown`→0(healthy)、`executed`→4(dead) |
| 4–7 | hooks | 0/1/2 | p1~p4 上钩次数 |
| 8 | gens | -1..5 | 剩余发电机数；无 HUD 时 -1 |
| 9 | t_half | 整数 | 时间（**半秒**单位），由帧名解析（`parse_time`）：`frame_MM_SS.5` → `MM*120+SS*2+1` |

- 由 `dataset_encoder.compact_frame` 从逐帧 CSV 编码；`dataset_encoder.frames_to_features` 可还原为旧版 30 维 one-hot（供模型/兼容）。
- 旧 30 维浮点表示已废弃（`features` 键移除）。

## 4. 时间语义

- 时间存于帧元组第 9 位（半秒整数）：从帧名解析 `frame_MM_SS.0.jpg` → `MM*120+SS*2`；首帧时间随实际时间轴
- 抽样间隔：0.5s/帧（产线 `run_video_*`）；旧测试帧目录为 10s/帧

## 5. 结局标注（label）

- `label` = 该局逃生人数 0–4（用户确认口径：看**最后一帧 HUD** 的 4 个幸存者头像图标，数 `escaped` 个数；`executed`/`dead`/`hooked`/`dying` 均不算逃生）
- **自动标注**：`labeling.infer_label_from_csv(csv)` 取该局末尾 `window=40` 帧，统计出现 `escaped` 的人数；窗口内无有效 HUD 时返回 -1
- 剪辑向/非单局片段（如 BV1aat）不进训练集，`label=-1`
- 历史种子标签已被证伪：原 BV1Uu=3、BV16=1 不正确（见 PROGRESS §3.22），已按真值修正
- `title`/`url` 在产线运行时自动填充（见下）

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

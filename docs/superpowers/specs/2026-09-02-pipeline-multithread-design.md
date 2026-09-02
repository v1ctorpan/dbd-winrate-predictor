# 多线程数据产线（下载→抽帧→检测→编码）— 设计文档

> 日期：2026-09-02
> 状态：已获用户批准（brainstorming 流程）

## 1. 背景与目标

项目已有链式工具：`yt-dlp`(下载) → `extract_frames.py`(抽帧) → `make_report.py`(逐帧 HUD 检测出 `detect_report.csv`) → `dataset_encoder.py`(编码进 `dataset/videos.jsonl`)。三者串行执行且互不感知，抽帧全程阻塞等待、检测要等全部帧落盘才启动。

目标：把产线改造成**同进程多线程流水线**，让抽帧与检测"边产边消费"，并以单条命令 `run_pipeline.py` 驱动从视频到 `dataset/videos.jsonl` 的完整转换。本次用真实视频 **BV1Z58J6bEoi** 跑通验证。

**关键约束**（已探查确认）：
- 检测侧 `make_report.py` 是"一次性扫描整目录"逻辑，需改造为可逐帧喂入的状态机才能流式。
- 帧文件名 `frame_MM_SS.0.jpg` 只有整秒精度；0.5s 抽帧会重名，需支持半秒命名。
- 视频可能含多个对局；对局切换时需重校准（healthy 基线/hook 槽位）并**分号归档**。

## 2. 用户确认的关键决策

| 项 | 决策 |
|---|---|
| 并行模型 | 同进程多线程流水线交接（不派发子 agent），每阶段一个线程 + queue |
| 交接粒度 | 逐批流式交接：抽帧线程每攒一批帧就通知检测线程 |
| 抽帧间隔 | 0.5s/帧 |
| 检测驱动 | 每有新帧就检测锚点；**有锚点才开始检测记录，无锚点继续等新帧**（自然跳过菜单/加载/结算段） |
| 多局处理 | 检测到换局信号（gens 非5→5 / 角色跳变）→ 收尾当前局、重校准、开新局记录 |
| 数据集多局归档 | `id` 带局号后缀：`BV1Z58J6bEoi:1`、`:2`… |
| healthy 基线 | 首个锚点后取连续相似（≥0.7）稳定帧建立（流式版 pick_opening_frame） |
| 本次验证 label | 先占位（-1），事后人工查看结算画面填入 |
| 交付物 | `run_pipeline.py` + 检测流式化改造 + 用 BV1Z58J6bEoi 跑通验证；流程记入 PROGRESS/文档 |

## 3. 架构设计

### 3.1 线程流水线

```
下载(一次性, yt-dlp, 手动/可选) 
   │  picture/raw_videos/{BVid}.mp4
   ▼
┌─ 抽帧线程 ──────────────────────────────┐
│ cv2 顺序读 mp4，每 0.5s 写 1 帧 jpg 到   │
│ picture/{BVid}/match_N/{frame}.jpg      │
│ 每攒 B 帧(如 50) 入队通知检测线程        │
└──────────────┬──────────────────────────┘
               ▼ (帧批次)
┌─ 检测线程 (流式状态机) ─────────────────┐
│ 每帧: 锚点检测 → 状态机推进              │
│ 当前局检测值累积 → 局结束写              │
│ report/{BVid}/match_N/detect_report.csv │
└──────────────┬──────────────────────────┘
               ▼ (完成的局)
┌─ 编码线程 ──────────────────────────────┐
│ dataset_encoder 增量 append 一行到       │
│ dataset/videos.jsonl (id 带局号)         │
└──────────────────────────────────────────┘
```

- 主线程驱动：`run_pipeline.py [video.mp4] [BVid]`，默认 `interval=0.5`。
- 队列用 `queue.Queue`，线程结束用 `Event`；抽帧线程 EOF 后广播"流结束"。

### 3.2 帧命名（0.5s 精度）

- 整秒帧：`frame_MM_SS.0.jpg`（向后兼容现有 parse_time）
- 半秒帧：`frame_MM_SS.5.jpg`
- `parse_time` 扩展解析 `.0`/`.5`，返回秒 + 半秒（float 或 秒*2 整数，spec 定整数半秒避免浮点）
- `features[29]` 时间分量随之为原始秒（含 0.5s 粒度），不归一化（沿用已定格式）

### 3.3 检测线程状态机（make_report 流式化核心）

```
WAIT_ANCHOR
   每新帧 detect_anchor；None → 丢弃继续等
   有锚点 → 记 anchor/scale，进 CALIBRATE
CALIBRATE
   攒帧：(a) 找连续相似≥0.7 稳定帧建 healthy 基线（流式 pick_opening）
        (b) 攒够开局若干帧校准 hook 槽位（calibrate_hook_slots 流式版，
            分批累积同帧共现列）
   校准完成 → RECORD
RECORD
   逐帧：classify 4 头像 / count_all hooks / GensTracker.update
        累积当前局行
   换局信号（gens 从非5跳回5，沿用 _detect_match_swaps 判据的流式版）
        → MATCH_END
MATCH_END
   收尾当前局：写 detect_report.csv → 通知编码线程 → 重置 → CALIBRATE
```

- 现有 `make_report.process_video` 保留或抽函数；新增一个 `streaming` 入口供流水线逐帧喂入，避免破坏现有 25+ 测试与旧调用。

### 3.4 目录布局

```
picture/raw_videos/{BVid}.mp4
picture/{BVid}/match_1/frame_*.jpg      # 每局独立目录
picture/{BVid}/match_2/frame_*.jpg
report/{BVid}/match_1/detect_report.csv
report/{BVid}/match_2/detect_report.csv
dataset/videos.jsonl                     # id=BVid:N, label=-1 占位
```

### 3.5 数据集编码

- `dataset_encoder.encode_csv(csv_path, video_id, label)` 已适配单文件聚合；流水线侧新增：`append_record(videos_path, record)` 幂等追加（行级增量写，读-改-写整文件 vs 直接 append 一行：选择直接 append，因每行独立）。
- `id` = `{BVid}:{match_no}`；`label` = `-1` 占位（人工回填后重写该行）。

## 4. 验证

- 用 BV1Z58J6bEoi（已下载，1080p，探查确认 HUD 与 BV1 同布局 scale=1.3，现有 config 可直接套用）。
- 跑通判定：抽帧线程产出全部帧（0.5s×~1115s≈2200 帧）、检测至少产出 1 局完整 detect_report.csv、dataset/videos.jsonl 新增带局号记录。
- 帧数/对局数打印在结尾汇总。

## 5. 待办边界（不在本 spec）

- 结算画面自动标注 label（仍人工）
- 胜率预测模型训练/推理（已有独立 plan）
- 真实下载并入产线脚本的自动化选项（本次下载已手动完成）

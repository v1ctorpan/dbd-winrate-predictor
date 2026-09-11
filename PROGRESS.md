# DBD 胜率预测项目 — 进展与设计文档

> 最后更新：2026-09-11
> 状态：HUD 区域校准完成；头像状态识别、hook 计数、发电机剩余数识别均对测试数据 100% 正确；发电机数字识别为**通用模板库 + 时序状态机（GensTracker）**；**全量 UT 提速 ~10x（583s→~57s，§3.15，现 113 tests）**；多线程产线 Task1-4、Task5 修复、BV1QUt 验证驱动修复 + match 级并行（§3.16）、BV1pht 真值纠正（§3.18/§3.19）、**自动锚点/切局可靠性修复（§3.21）**、**结局自动标注 `labeling.py` + 高亮逃出识别（§3.22）**、**`videos.jsonl` 紧凑帧元组存储（§3.23）** 均已推送。dataset 现 7 行（BV1Uu 2 局、BV16=3、BV1pht=1、BV1QUt=0/0 已自动标注；BV1aat=-1、BV1Uu m1=-1 不入训练）——**均已 0.5s 密集重跑、每局 t 从 0**（2026-09-11 重建，见 §3.5）。剩余：gate_ui、数据积累等（见 §6、§3.17~§3.23 注意点）；**对局序列管道已实现并推送（§3.5，130 tests PASS）**。

## 分支与提交状态（2026-09-08）

- `main`（当前分支）：Task1-4（`80bdf87`~`83deb92`）与 Task5 修复均已推送。本会话推送（时间序）：`6f5dd84` UT 提速 + BV1aat 合并、`c450619`/`7a203ab` BV1QUt 修复 + 并行产线、`719e6b1` executed 状态、`b0ef16a`/`7446bd3` BV1pht 全片/单局纠正、`54e51bd` 流程 profile 提速、`ec9b98f`/`062206d` BV1QUt 两局、`5d9be9c` run_video 统计 + 人工单局入口、`540a9b2` 自动锚点/切局可靠性、`84d248e` 结局自动标注、`813e7de` BV1Uu 重分段、`ee853c3` 高亮逃出识别、`411f2a9` 存储优化。全量 **113 tests ~57s PASS**。
- `data_pipeline`：已合入 main，本地残留分支可删（`git branch -d data_pipeline`）。
- 产物：BV1pht96fEjN.mp4（1080p 14.3min 435MB）、BV1QUt766Etg.mp4（1080p 21.4min 641.8MB）均在 `picture/raw_videos/`（gitignore）。BV1pht 正式 report 只保留 `match_1`（1690 帧）；BV1Uu 正式 report 有 `match_1`/`match_2`（重分段）。
- **整体待办见 §6；注意点见 §3.17**。

## 0. 基础要求
- 请使用中文进行对话，在compaction中要显式提到这一点
- 耗时较长的操作提前告知用户并确认，尽量采用subagent后台运行

## 1. 项目目标

- 输入：一局 DBD 的录播视频
- 处理流程：
  1. 视频抽帧（建议每 0.5s 一帧）
  2. HUD 识别（传统 CV + OCR，不用训练识别模型）
  3. 每帧生成结构化状态向量（4 名幸存者状态/上钩次数、剩余发电机数、大门状态、时间）
  4. 从结尾结算画面自动标注本局结局（逃生/死亡）
  5. 积累带标注数据集
  6. 训练机器学习模型
  7. 输出整局双方胜率走势图
- 用户选定的关键决策：
  - 使用形态：**离线复盘分析**（非实时覆盖层）
  - 预测方法：**机器学习模型**（先积累带标注数据）
  - 结局标注：从结算画面**自动标注**
  - HUD 识别：**传统 CV + OCR**，不训练识别模型

## 2. 架构与核心设计

### 2.1 锚点方案（关键）

HUD 大小会随玩家分辨率/缩放变化，因此采用"锚点"确定缩放：

- 锚点 = 发电机图标 `picture/gen.jpg`（模板 35×32）
- 检测：多尺度模板匹配 + 聚类，取**尺寸最大**的图标
  - 场景中可能有多个相同图标（幸存者动作进度），一律忽略，只取最大且靠左下角的
- `scale = 匹配尺寸 ÷ (35×32)`，一局开头确定一次、全程缓存复用，之后不重新检测
- 阶段判定：
  - 找不到锚点且已确定过 scale → 局中无图标帧（发电机全部修完）
  - 从未确定 scale → 局前/加载帧
- 所有 HUD 区域在配置中以**锚点相对坐标**存储，运行时用 `rel_to_abs` 换算

### 2.2 帧语义约定

- 时间**不做 OCR**，从文件名解析：约定 `frame_10_13` = 10 分 13 秒
  - 注意：当前 `picture/test1` 的示例帧（`frame_0000`~`frame_0011`）是手工命名的序号，未带该时间逻辑
- 结局标注：从结算画面自动标注（尚未实现）

### 2.3 区域配置

- 配置文件：`config/hud_regions.json`（锚点相对坐标）
- 锚点：绝对坐标 `(94, 536)`，w=35 h=32，scale=1.0，score=0.9448
- 区域（绝对坐标，scale=1.0 时）：
  - 头像 `survivor_p1~p4`：x=62–110，宽48，高44；顶部 y = 282 / 341 / 400 / 459（步长59）
  - 钩子 `hook_p1~p4`：x=146–169，宽23，高20；顶部 y = 283 / 342 / 401 / 460（步长59）
  - 发电机 `gens_row`：rel (-29,-4) → (38,35)
  - 终局倒计时 `endgame_timer`：rel (134,-532) → (951,-494)（frame_0011 校准）
  - 大门 `gate_ui`：rel (-31,-8) → (10,32)（frame_0011 校准）
  - 地窖 `hatch_ui`：rel (21,-8) → (57,34)（frame_0011 校准）
- hook_p1-4 在 frame_0008 校准；endgame_timer / gate_ui / hatch_ui 在 frame_0011 校准
- 头像/钩子区域已程序化对齐（`align_regions.py`）

## 3. 当前进度

### 3.1 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| 锚点检测 | `hud_anchor.py` | ✅ 验证通过：frames1–9 稳定命中 (94,536) scale=1.00（score 0.77–0.94）；聚类可合并同图标 1.1 缩放噪声 |
| 区域坐标转换 | `hud_regions.py` | ✅ |
| 交互式校准工具 | `hud_calibrate.py` | ✅ 鼠标拖框校准，`TRUST_SCORE=0.80` 只接受可信锚点 |
| 区域对齐 | `align_regions.py` | ✅ 头像/钩子对齐到均匀网格 |
| 头像状态识别 | `state_recognizer.py` | ✅ **48 格全部正确**（健康/受伤/钩/倒/死/逃） |
| hook 计数 | `hook_counter.py` + `calibrator.calibrate_hook_slots` | ✅ **720p 48 格 + 1080p 28 格全部正确**（0/1/2 道白线，槽位每视频自动校准） |
| 发电机剩余数 | `gens_counter.py` | ✅ **test1 12 帧 + BV1 110 帧 + BV16 30 帧全部正确**；通用模板库 + GensTracker 时序识别（数字 1~5 / 图标消失=0 / 无 HUD=None） |
| 对局序列管道 | `match_dataset.py` / `match_model.py` / `train_sequence.py` / `predict_live.py` | ✅ 读 `dataset/videos.jsonl` 紧凑帧元组 → 30 维特征（时间每局归零/600s）；GRU 变长训练 + stateful 逐帧推理跑通；数据仅 5 局故仅 smoke test |
| 验证辅助 | `apply_regions.py` / `annotate_regions.py` / `extract_crops.py` / `make_montage.py` / `make_hook_montage.py` / `find_gen.py` / `find_gen_multi.py` / `fix_anchor.py` | ✅ |

### 3.2 头像状态识别细节（`state_recognizer.py`）

- 图标类状态（hooked / dying / dead / escaped）：模板匹配 NCC，阈值 0.55，全部命中
- 健康 vs 受伤：曾尝试启发式（diff vs 健康参考、对角投影、Hough 直线）**全部失败**
- 最终方案：**红色调覆盖层特征**
  - 受伤头像带红/绛红覆盖 → 红度和饱和度显著升高
  - 判定：`R-G` 与饱和度相对该槽位"健康基线"的差值超阈值（`inj_rg_delta=12`，`inj_sat_delta=15`）
  - 用开局全健康帧建立每个槽位的健康基线，跨角色更鲁棒
- frame_0000 无 HUD → 正确判为 unknown（NCC 低于 face_thr 0.35）

### 3.3 hook 计数细节（`hook_counter.py`）

- 显示规则：钩子区有 2 个固定槽位（720p：绝对 x≈155/160 = crop 内 9/14；1080p：绝对 x≈193/200），白竖线=已钩、黑竖线=未钩
- 判定为"线"需同时满足：
  1. 只在**校准槽位 ±1** 内检测（不再整区域扫描，避免区域边框/噪声列干扰）
  2. 纵向**连续 ≥4 行**亮像素（排除孤立斑点；1080p 噪声 run=3 被此过滤）
  3. 高于该 crop **背景中位数 + 12**（自适应亮度，720p 与 1080p 通用）
  4. 最长亮段**延伸到 crop 底部 40% 以内**（1080p 区域顶部噪声 run≥4 但到不了底部，被此过滤）
- 槽位来源：`calibrator.calibrate_hook_slots` 按帧频自动校准
  - 方法：全帧扫描区域中"高于 bg+12、run≥4、达底部 40%"的竖列，跨 4 槽位/全帧聚合计数，取前 2 个互距 ≥4 的峰值列
  - 720p 实测校准 → [155,160]（12/12 正确）；1080p → [194,201]（7/7 正确）
- 关键发现：**hook 槽位不与锚点等比缩放**（相对锚点偏移 61→72，仅 1.18×，而头像/区域整体为 1.31×）。按锚点比例推断的 1080p 槽位应为 [200,207]，实测为 [193,200]。故槽位必须**每视频单独校准**，不能按锚点 scale 直接外推
- 统一的计数算法（bg+12 / min_run=4 / bottom_frac=0.4 / slot±1）**不再需要每视频不同的阈值**——此前 720p 需 bg+12 而 1080p 需固定 150 的矛盾由底部约束 + min_run=4 解决
- 裕度扫描验证：阈值在 5~20 范围内全部正确（取 12，两边各留 ≥7 裕度）
- 测试：`tests/test_hook_counter.py`（720p 12 帧 + 1080p 7 帧真值断言 + 自动槽位校准断言，unittest）

### 3.4 gens 剩余数识别细节（`gens_counter.py`）

- `gens_row` 区域实际显示"剩余台数数字(1~5) + 发电机图标"；全部修完时变为逃生大门图标
- 三态判定：
  1. 区域亮像素过少（`(g>100).sum()<20`）→ `None`（开局 HUD 未加载，frame_0000）
  2. 发电机图标模板匹配分数 `≥0.70` → 读左侧数字（NCC 分类）返回 1~5
  3. 否则（图标消失）→ `0`（全部修完，大门状态，frame_0010/0011）

#### 通用数字模板库（asset/gens_digits/）

- **背景**：旧方案 `count_gens()` 每帧独立匹配，且依赖 per-video 手动指定参考帧（BV1/BV16 硬编码在 `make_report.py`），无法用于后续**实时间隔截图**场景（不可能提前看完视频构建模板）。
- **spike 验证**：test1(720p) 与 BV1/BV16(1080p) 的数字渲染差异巨大（720p 粗体 vs 1080p 细体，NCC 仅 0.60-0.68），**静态单一模板不可行**。但**高清多视频模板库**可跨视频通用：
  - BV1 模板识别 BV16（放大到 scale=1.5）：全部正确
  - BV1 模板识别 test1（缩小到 scale=1.0）：全部正确
- **方案**：从 BV1(1080p) + BV16(1080p) 提取各数字样本存入 `asset/gens_digits/`（当前 45 样本，`digit_N_MM.png` + `refs.json` 索引），启动时 `load_digit_refs()` 加载。
- **模板迭代**：新视频某数字匹配不佳时，将人工核实的数字帧追加到模板库（按 `digit_N_MM.png` 命名 + 更新 refs.json）即可迭代逼近通用。实测：BV16 的 5 在仅有 BV1 模板时与 3 混淆（0.59 vs 0.60），追加 BV16 的 5 样本后分数升至 1.0。

#### GensTracker 时序状态机

`count_gens()` 是无状态纯函数，单帧独立匹配有局限（如 test1 frame_0007/0009 的 2 分数仅 0.51/0.37 < 0.55 阈值会误判 None）。新增 `GensTracker` 类利用 gens 数字的**时序单调递减**特性提升鲁棒性：

```
每帧 update(frame, resolved, anchor)：
1. 帧间沿用：与前一帧数字框全像素 NCC ≥ 0.85 → 沿用前一帧结果（免模板匹配）
2. 模板重识别：高清模板 resize 到当前帧数字框尺寸求 NCC，取最高分
3. 递减约束：识别结果 > 前一帧 → 沿用前一帧（gens 只减不增，防御误报）
4. 低置信：best ≥ LOW_THR(0.45) 且符合递减 → 采纳；否则沿用前一帧有效数字（防御渲染噪声）
换局时调用 reset() 清空状态（prev_digit/prev_crop）
```

- 参数：`GEN_ICON_THR=0.70`、`DIGIT_THR=0.55`、`LOW_THR=0.45`、`TRACK_NCC=0.85`、`DIGIT_X=29`（×scale）
- 实测效果：BV1 110 帧**无任何 None**（旧方案 07_40/08_10 因分数不足误判 None，tracker 时序沿用解决）；test1 12 帧全对
- `make_report.py` 已删除 BV1/BV16 硬编码数字映射，统一走模板库 + tracker；换局帧（gens 从非 5 跳回 5）reset tracker
- 设计文档：`docs/spec_gens_tracker.md`
- 测试：`tests/test_gens_tracker.py`（test1/BV1/BV16 全帧真值断言 + reset 测试）

#### 真值修正

- **BV16 frame_03_40/03_50 真值为 1 而非 2**：旧报告用 03_40 自身作 2 的模板（循环论证）误判为 2；列投影对比 BV1 数字 1 吻合。正确序列 2(02_20~03_30)→1(03_40~03_50)→0(04_00+)，2→1→0 平滑递减。
- BV1 第 1 局正确序列：5(00_00~01_20)→4(01_30~03_20)→3(03_30~05_30)→2(05_40~09_00)→1(09_10~09_30)→0(09_40~10_30)，frame_10_40 换局。

### 3.5 对局序列数据管道与结局预测模型（已实现，2026-09-11）

把逐帧检测值累积为**带结局标签的变长序列数据集**，训练模型预测结算结局。设计文档：`docs/superpowers/specs/2026-09-02-match-sequence-design.md`，实现计划：`docs/superpowers/plans/2026-09-02-match-sequence.md`。

**用户确认的关键决策**：

| 项 | 决策 |
|---|---|
| 预测目标 | 结算结局 = 逃生人数 0-4（5 类多分类） |
| 训练/推理模式 | 训练用完整序列→结局；推理实时逐帧用前缀序列预测（每帧刷新） |
| 数据规模 | 上千局带标注 |
| 数据格式 | 一局 = 一个 JSONL（每帧一行 30 维特征 + 末行 label）；`dataset/matches/{match_id}.jsonl` |
| 特征 | 30 维定长：4 人状态 one-hot(6类×4=24) + hooks(4) + gens(1) + 归一化时间(1) |
| 模型 | GRU 序列分类器（input=30, hidden=64, 5 类 softmax） |
| 训练 | 随机截断前缀采样（对齐实时前缀分布）+ pack_padded_sequence 处理变长 |
| 实时推理 | Stateful GRU：hidden 跨帧累积，逐帧单步 forward，毫秒级 |
| 数据切分 | 按局切分（不按帧），train/val/test 80/10/10 |
| 技术栈 | 新增 PyTorch（当前已装 2.2.2 但 dylib 缺失，实现时需修复） |

**文件规划**：`dataset_encoder.py`（CSV→JSONL）、`match_dataset.py`（变长批次+截断采样）、`match_model.py`（GRU）、`train_sequence.py`（训练）、`predict_live.py`（实时推理）。

**种子数据**：BV1 结局 label=3、BV16 结局 label=1（人工标注）；test1 为 720p 随机采样帧，不作序列样本。

**实现进展（2026-09-11）**：
- **与计划的偏差（重要）**：存储已改为 `dataset/videos.jsonl` 紧凑帧元组（不再是 per-match 30 维 JSONL）。因此跳过计划的 Task1（编码器已有 `dataset_encoder.frames_to_features`）；`match_dataset.py` 改为**直接读 `videos.jsonl`**、过滤 `label<0`、用 `frames_to_features` 还原 30 维，并把**时间每局归零后 /1200（半秒单位，即 /600s）**归一。
- 新增文件：`match_dataset.py`（MatchDataset/trucated_item/collate_fn）、`match_model.py`（MatchGRU：变长 pack_padded + 单步 `single_step`）、`train_sequence.py`（按局切分 + 随机截断前缀训练 + ckpt；已修「val_acc=0 时不保存 ckpt」缺陷）、`predict_live.py`（stateful 逐帧推理，时间归一与训练一致）。`requirements.txt` 记录依赖。
- 环境：dbd env 现为 **Python 3.12.14 + torch 2.2.2+cpu（可 import，无 CUDA）**；计划里的「dylib 缺失修复」已过时。
- 全量 **130 tests PASS**（原 113 + 新增 16）。
- **注意**：`videos.jsonl` 可用（label≥0）仅 **5 局**且类别偏斜（0×2/1/2/3），训练仅为**流程 smoke test**（val_acc 恒 0，ckpt 由 best 初值 -1 保证保存）；真正精度需等数据积累到上百~上千局。
- **每局时间归一为从 0 起**：`encode_csv` 以该局首帧为 t0，`compact_frame(row, t0)` 输出 `t-t0`；已一次性迁移现有 `videos.jsonl`（各记录首帧 t=0）。`match_dataset`/`predict_live` 的时间分量仍按「每局归零 ÷600s」归一，二者一致。
- **数据集 0.5s 重建（2026-09-11）**：5 支视频全部按 **0.5s** 重跑（BV16/BV1Uu 原为 10s 稀疏已补齐；BV1/BV1aat/BV1QUt 原已 0.5s）。命令：`run_pipeline.py <mp4> <bvid>` + `--single-match`(BV16/BV1aat/BV1pht) / `--prescan --parallel`(BV1QUt/BV1Uu)。重建后 7 行、每局 t 从 0；帧数：BV16=578、BV1aat=820、BV1pht=1707、BV1QUt=1243/1224、BV1Uu=1278/918。自动标签与原真值一致（3/-1/1/0/0/-1/2；`BV1aat`、`BV1Uu m1` 按既有约定恢复 -1）。源视频 `BV16/BV1aat/BV1QUt/BV1Uu` 已重新下载到 `picture/raw_videos/`（gitignore）。

### 3.6 多线程数据产线（设计定稿，实现进行中）

为把"下载→抽帧→检测→编码"串成自动化产线并支持多线程提速，设计已定稿：`docs/superpowers/specs/2026-09-02-pipeline-multithread-design.md`，实现计划 `docs/superpowers/plans/2026-09-02-pipeline-multithread.md`（5 任务 TDD，直接提交 main 无分支）。

- 抽帧时间精度升级：`parse_time` 支持**半秒精度**，帧名 `frame_MM_SS.0.jpg`（整数秒）/ `frame_MM_SS.5.jpg`（半秒）；`extract_frames.py` 的 `frame_name(t)` 与 `--interval 0.5` 兼容（commit `80bdf87`，全量 34 测试 PASS）。
- 检测侧复用既有函数：`make_report.pick_opening_frame/build_refs/classify/build_opening_refs`、`calibrator.calibrate_hook_slots`（路径版）、`gens_counter.GensTracker`、`hook_counter.count_all`；新增流式检测器状态机 WAIT_ANCHOR→CALIBRATE(budget=12)→RECORD，`apply_hook_cfg` 扩展支持 hook_names 列表。
- 编码侧：一局 = 一行 JSONL 追加进 `dataset/videos.jsonl`，`id="{BVid}:{match_no}"`，label=-1 待标注。
- 任务状态：Task 1（帧命名 + parse_time）✅ commit `80bdf87`；Task 2（流式检测器）✅ commit `f3de624`；Task 3（追加式编码）✅ commit `8f6f730`；Task 4（三线程 run_pipeline）✅ commit `83deb92`；Task 5（BV1pht96fEjN 端到端）⏳ 进行中。

### 3.7 WAIT 锚点误触修复（`3c55d96`，已并入 main，2026-09-03）

**现象**：对 BV1pht96fEjN.mp4 跑 0.5s×前 90s 小样本，产出全垃圾——scale 卡 0.40、p1-p4 几乎全 unknown、gens 全 None，整段被当 1 局。

**根因**（逐层排查，探针先用 BV1 已知帧验证了方法可信）：
1. 视频开头 ~40s 是菜单/过场（无 HUD），但 `WAIT_ANCHOR` 用**单帧 no-prior `detect_anchor`（取全局 max-score）**。
2. 菜单帧里 scale≈0.4 的小尺度误报得分 0.79–0.82，**比真实 HUD 发电机图标（0.70–0.79）还高** → 第 0 帧即命中误报 `(1474,739)@0.40`，触发 `match_no=1`。
3. 之后以该错误位置+尺度为先验锁定，真实图标在 `(142,806)@1.5` 永远对不上 → 整局分辨率全错。
4. 真实对局里单帧 no-prior 也会被 0.4 尺度噪声压过，故不能只信单帧。

**修复**（`stream_detector.py`）：`WAIT_ANCHOR` 不再单帧触发，改为**滑动窗口共识** `_wait_anchor(frame)`：
- 每帧取 `find_gen_anchors`（保留**所有**候选簇，而非 max-score 单点），记录进 `_wait_cands`（最近 `wait_window`=6 帧）。
- 候选需 `scale >= wait_min_scale`(0.9)（排除菜单 0.4-0.9 噪声），并按位置(±15px)+尺度(±0.25)跨帧聚类。
- 同一位置簇在最近窗口中 ≥`wait_min_frames`(3) 帧出现**且含当前帧**才开局。
- 实测：菜单 t=0-4s 保持 WAIT；对局段 t=116-121s 正确转 CALIBRATE → RECORD；样本 CSV 出现合理 p 状态演化（healthy→injured→dying），用户确认区域对齐准确。
- 构造参数：`StreamingDetector(..., wait_window, wait_min_frames, wait_min_scale)`。

**新增测试**：`test_wait_requires_stable_position_across_frames`——6 帧不同位置粘贴图标（模拟菜单抖动）应保持 WAIT；连续 4 帧同位置粘贴应转 CALIBRATE 且 `match_no==1`。全量 41 测试 PASS。

### 3.8 Task5 排查：BV1pht96fEjN gens≈0 / hooks≈0（2026-09-04，已修一部分）

**复现**：对 `picture/raw_videos/BV1pht96fEjN.mp4`（1080p 30fps 14.31min）前 ~280s 跑 `run_pipeline.py --sample 560`，得到 gens 312/338 帧 =0、hooks 全 0。

**根因 A（gens 阈值，已修）**：`GEN_ICON_THR=0.70` 太高。本视频 gen 图标 NCC 恰在 0.66~0.81 波动，多数帧 <0.70 走"图标消失→0"；且 prev=0 后被单调约束锁死回不到 5。跨视频实测：**有 gen 图标帧 NCC≥0.66（BV1pht 最低）、"全修完/大门"帧 NCC≤0.28**（test1 0.24~0.28 / BV1 0.16~0.27），判隔巨大 → `GEN_ICON_THR` 降到 **0.55**。新增回归测试 `test_borderline_gen_icon_still_reads_digit`（用本视频 gens 区域裁剪 fixture）。修复后回放：0 帧变 5；560 帧样本 gens 正常演化（5→…→3）。

**根因 B（hooks 槽位校准时机，已修）**：`stream_detector` 只在开局前 12 帧一次性 `calibrate_hook_slots`；开局无人上钩→空槽位→永远 0。改为**前向滚动重校准** `_maybe_recalibrate_slots`：RECORD 中持续缓冲最近 `slot_win`(40) 帧路径，检测到竖线候选帧时重试校准，锁定后停止。新增回归测试 `test_hook_slots_recalibrated_after_late_hooks`。TDD：先 RED 后 GREEN，全量 43 测试 PASS（原 41 + 2 新）。

**用户目视真值修正（frame_04_20.0~04_29.5 段）**：
- `frame_00_22.5`：p4 **hook=0**（CSV 曾 1，误检）；`frame_00_23.5`：p1 **hook=0**（CSV 曾 2，误检）
- `frame_04_00.0~04_07.5`：p3 **hook=0** 且 dying；此后她被执行（**executed ≈ sacrificed ≈ dead**，处决状态代码尚未补充）→ 该段 CSV `0/0/2/0` 为**误检**（dying 段出现类竖线干扰）
- `frame_04_20.0~04_29.5`：画面正常，**gens 实为 4**（CSV 曾 0/None/4 → 该段 0 为误读）
- `frame_00_55.5`、`frame_02_10`：**HUD 真实短暂消失**（gens=None 正确）→ 后续宜复用前一帧状态，但须确认是真消失而非误检

**残留待办**：① dying/executed 段的 hook 竖线误检（dying 状态指示 UI 与钩子线混淆）；② ~~260~269s gens 误读 0（图标判读边界）~~ ✅ 已修（§3.11）；③ HUD 短暂消失时"复用前状态"实现 + executed≈dead 状态补充。

### 3.9 残留 A 排查进展：hook 误检 = overlay 亮带/闪现（2026-09-04，部分已修）

**关键真值（用户目视确认）**：**BV1pht96fEjN 全片（至少 0~280s 采样段）自始至终无人上钩**——屠夫用处决(Mori)击杀倒地者，从不挂人。故该视频 **hooks 真值恒为 0**；一切非 0 读数都是误检。dying 段（如 `frame_04_00.0~04_07.5` p3）即此前误报 `0/0/2/0` 的来源。

**框选核对（用户目视）**：`config/hud_regions.json` 派生出的 hook 区域框（BV1pht 绝对 x220~254）**位置正确**；区域内平时是**两根黑色竖线 = 未挂(hook=0)**，挂人才变白。

**像素形态分析**：
- 真钩模式（test1/BV1 实测）：每根钩 = slot±1 的 3 列亮簇；双钩 = 两个独立簇、**簇间空隙不亮**。例 BV1 p2 双钩 lit=[193,194,195,200,201,202]。
- 误检模式：① overlay 亮带（受伤/追击/处决等 UI 盖过 pip 区）成**单一连通亮带贯穿两槽**（如 BV1pht lit=[222..228] 或 [223,224,225]）；② 孤立窄簇闪现（如 {222}、{222,223}，与真单钩像素不可区分）。
- 反例需容忍：test1 p4 双钩帧空隙会有 1 个孤立亮列（lit=[154,155,156,157,160,161]），不可当 overlay 误拒。

**已修：`hook_counter.count_hooks` 防 overlay**——若区域内存在单一连通亮带同时贯穿两个槽位邻域 → 判 0（TDD：`TestCountHooksAntiBlob`，先 RED 后 GREEN）。720p/BV1 真值测试全部保持通过。全量测试 46 PASS（43 + 3 新）。**但 BV1pht 端到端全 0 尚未最终验证**：孤立单簇闪现(2)型仍有残余（计划以"持久化地板"：真钩一旦亮起会持续 ≥K 帧才抬升输出，闪现不采纳；顺带实现 HUD 消失时复用前值）。滚动校准仍有在开局误锁 overlay 列成槽位的风险，需配合确认/支持度门限，属后续项。

### 3.10 接手须知 / 特别说明（2026-09-04）

- **本机/仓库状态**：main 已含 Task1-4 + WAIT 修复（`a0b1674`）并推送；Task5 排查修复（gens 阈值 + 滚动校准 + hooks 防 overlay）在本分支**尚未提交**，本次随 PROGRESS 一并提交。
- **`picture/raw_videos/BV1pht96fEjN.mp4`** = 已用 yt-dlp 重下（1080p 30fps 14.31min，435MB，`dbd` env 内 `python -m yt_dlp`；无 cookie 可用 1080p30 格式 id 30080）。**勿入库**（已 gitignore `picture/raw_videos/`）。
- **任务产物不入库**：`picture/BV1pht96fEjN/match_1/`（549 帧样本，含 `_diagA/` 标注图）与 `report/BV1pht96fEjN/`（嵌套 CSV）为调试输出，**不要 git add**；仅两个 fixture `picture/BV1pht96fEjN/gens_borderline_{a,b}.png` 需入库（回归测试引用）。
- **关键域知识**：BV1pht96fEjN 里屠夫**从不挂人（Mori 处决）** → 全片 hooks 真值 0，不能用来标定钩子槽位；"executed(处决)" 未编码，语义 ≈ sacrificed ≈ dead；dying(倒地) 的出血/UI 会盖过 pip 区产生类钩线。
- **测试运行**：`& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`（当前 46 PASS）。
- **遗留未决点（接手重点）**：① ~~持久化地板以消残余单簇误检 + HUD 消失复用~~ ✅ 已修（§3.12）；② ~~滚动校准防误锁槽位~~ ✅ 已修（§3.12）；③ ~~gens 260~269s 误读 0（真值 4）复核~~ ✅ 已修（§3.11）；④ executed≈dead 状态补充；⑤ 全片端到端(14.3min, ~55min)与数据校验。

### 3.11 gens 260~269s 误读 0 根因与修复（2026-09-06，已修）

**现象**：BV1pht96fEjN `frame_04_20.0~04_29.5`（260~269.5s）真值全 4，CSV 曾全 0。

**根因（两项叠加）**：
1. **单帧伪 0**：`frame_04_20.0`（260.0s）恰是 5→4 完成瞬间，HUD 渲染白光扰动使 gen 图标 NCC=0.472 < `GEN_ICON_THR`(0.55)、digit 4 也跌到 0.438 < `LOW_THR`(0.45)。该帧被误入"图标消失→0"分支。
2. **prev=0 单调锁死**：`GensTracker`"只减不增"守卫（`best > prev_digit → 沿用 prev`）把 0 当永久地板——04_20.5 起每帧正确识别的 4（icon 0.757 / digit 0.783）都被压回 0，直到 04_29.0 出现 None 重置才恢复 → 整段 10s 全 0。

**修复（`gens_counter.py`）**：
- 新增 `LOW_ICON_THR=0.45`：图标 NCC **<0.45 才算"消失→0"**（跨视频实测真 0 帧≤0.28，判隔充足）；`[0.45,0.55)` 视为"HUD 扰动但图标仍在"，照常走数字识别/沿用前值，绝不判 0。
- 递减守卫仅当 `prev_digit ∈ (1..5)` 时拦截；**prev=0 不设地板**（真 0 状态图标已消失不会走到数字分支，prev=0 只可能来自误判，允许置信识别恢复）。
- 移除不再使用的 `icon_thr` 构造参数。

**验证（TDD，先 RED 后 GREEN）**：
- 新增回归测试 `test_transient_icon_dip_does_not_lock_zero`（用本视频保存的 fixture `picture/BV1pht96fEjN/gens_0420_zero.png`(04_20.0 区域) + `gens_0420_four.png`(04_20.5 区域)）：RED 时 second=0（锁死复现）→ GREEN 后 first≠0、second/third=4。
- 全量 51 测试 PASS（46 + 5 新增累积；含 test1/BV1/BV16 真值回归不受影响）。
- 固定锚点(142,806,scale1.5) 顺序回放真实帧：`04_15.0~04_19.5`=5 → `04_20.0`=5（5→4 完成瞬间白光帧，沿用前值，用户认可）→ `04_20.5~04_29.5` **全 4** → 之后持续 4。
- 遗留：`count_gens`（无状态，`make_report.py`/`detect_report.py` 遗留路径）仍用 `>=GEN_ICON_THR` 判图标，未同步改动；新产线 CSV 走 `GensTracker`，不受影响。

### 3.12 hooks 持久化地板 + 滚动校准跨帧支持度防误锁（2026-09-06，已修）

承接 §3.9 残留：孤立单簇闪现(2)型误检与开局 overlay 被锁成槽位。

**hook 持久化地板（新文件 `hook_persist.py`，`HookPersist`）**：raw hook 计数一次抬升必须连续 ≥k(默认 4) 帧都观测到才采纳，消除开局/转场的孤立单簇闪现（BV1pht hooks 恒 0 场景）；`hud_ok=False`（HUD 短暂消失/结算转场）帧不更新输出，复用前一稳定值。抬升与回落都需 k 帧确认（`test_release_needs_sustained_zero`）。`stream_detector._record` 接入：`raw_hooks = hook_counter.count_all(...)` → `hooks = self._hook_persist.update(raw_hooks, hud_ok=cur is not None)`；开局锚定与换局时 `reset()`。回归测试 `tests/test_hook_persist.py`（6 项）。

**滚动校准防误锁（`calibrator.calibrate_hook_slots` 加 `min_frames`）**：候选 (a,b) 槽位对必须出现在 **≥min_frames(默认 2) 个不同帧**才被采纳——旧逻辑只看累计共现次数，1~2 帧的 overlay 宽亮带一旦构成 best 对就会被误锁成槽位。改为按不同帧计数过滤后再取 best。回归测试 `tests/test_calibrator.py` `TestCalibrateSlotsSupport`（3 项：单帧/两帧 overlay 闪现不锁、真实双槽持续多帧照常锁）。

> 注：`tests/test_calibrator.py` 原 `TestConsensusAnchor`/`TestCalibrateVideo`（针对 `calibrator.consensus_anchor`/`calibrate_video`）随本次改写被移除——这两函数已被 `hud_anchor.detect_anchor` 取代（产线不再调用，属孤儿函数），暂保留未删。

**验证**：TDD 先 RED 后 GREEN；全量 51 测试 PASS。BV1pht 端到端 hooks 恒 0 的最终验证仍待 §6.8 全片回放。

### 3.13 新视频 BV1aatX6uE3C 端到端 + gens 帧间沿用 scale 抖动崩溃修复（2026-09-06）

**背景**：用新下载视频 BV1aatX6uE3C（标题「'对你没听错 审判者居然还加强'【黎明杀机】」，1080p30，418s，221MB，`picture/raw_videos/`，gitignore）做产线端到端验证并写 dataset。0.5s 采样冒烟（120 帧）**立即崩溃**：`gens_counter.GensTracker.update` 帧间沿用 `_ncc(digit_crop, prev_crop)` 时两帧 anchor scale 不同（BV1pht 锚点恒定所以从未暴露），数字框宽 43 vs 34 → matmul shape 不匹配 `ValueError`。用全片帧差探测确认该片为高能剪辑向（418s 内 814 次画面大跳），非单一完整对局。

**修复**：帧间沿用前先校验两 crop 尺寸，不一致则 `cv2.resize` 对齐再比 NCC（`gens_counter.py`）；模板重识别逻辑不变。新增回归 `test_anchor_scale_jitter_does_not_crash`（`tests/test_gens_tracker.py`，尺寸 0.8× 前帧 + scale 0.8 锚点，第二帧应沿用 4 不崩溃）。

**端到端结果**（全片 0.5s×836 帧，~21min 实跑）：产出 4 个自动分段 match（match_1~4），全部 encode+append 至 `dataset/videos.jsonl`（`BV1aatX6uE3C:1~4`，label=-1，features 30 维、时间列单调、1/2 帧无信息量但 match_3/4 为 2min/6min 连续段、含 gens 5→2/0 演化与 dying/injured 状态）。`run_video` 返回的 `closed=[]` 系统计口径缺陷：closed_q 已被 encoder 线程消费空后才统计行数，实际以 append 记录数为准。结论：检测器对剪辑向视频可分长段跟踪并自动分段，但此类视频非单局样本，标签数据仍须用完整单局视频。

**验证**：全量 52 测试 PASS（新增 1 项）。

### 3.14 换局误切修复 + 10s 预扫优化流程（2026-09-06）

**背景**：BV1aatX6uE3C（真值=单局至 ~6:45/~405.5s）在旧换局判定下被切成 4 段——旧逻辑 `_prev_g` 只在「gens==5 且上一帧是孤立 0」计数，误把 26.5/58.5/175.5s 的三处单帧 0 后接 5 判成换局。逐帧排查另确认：真实换局 BV1Uu frame_10_40 前有连续 0 段；BV1aat ~290s 处有 8 帧连续 None/0 死段（04_45.5~04_49.5）→ 修好单帧后仍会在 ~290s 再切一次（残留，见下文处理）。据此用户定方向：**不做全量全片重跑，改为实现“10s 预扫优化流程”**。

**修复（`stream_detector.py`，换局判定收严）**：`_prev_g` 改 `_dead_run`，加 `MIN_END_ZERO_RUN=2`——0/None 帧累计、其他值清零、仅当「识别值==5 且连续死段 ≥2」才切局；孤立单帧 0 不再切。新增回归 `TestMatchEndRegression` 2 项（孤立单帧 0 不切 / 连续 0 段仍切，全过）。

**预扫设计（`prescan.py`，新文件）**：对任一新视频先按 10s 间隔抽样，
- **确认 HUD 位置**：逐样本全帧 `find_gen_anchors` → 跨样本位置/尺度聚类取最大簇为共识锚点 `anchor`，簇占比 `anchor_ratio`；
- **判多局**（用户定的双条件，须同帧满足）：A) gens 异常跳变——识别值高于此前递减基线（或同值但中间隔着 ≥2 个不可读样本 / 识别到 0）；B) 4 个幸存者头像与前序含头像样本的 **NCC 均值 < 0.8** 判为“明显区别”。A∩B 才切局，只满足一个不切。
- 实现分层：纯决策 `find_boundaries`（可单测）+ 抽样层（stateless `count_gens` 而非 GensTracker——后者单调地板会掩盖跳变；4 头像裁剪灰度 std 校验 + NCC 均值）→ `PrescanResult`。

**调试要点（BV1Uu 实测从 3 个假界收敛到 1 个真界）**：首版把「同值 + 单样本 None 断档」也当候选，配合局内头像 NCC 偶发跌到 0.69~0.75（同批幸存者的状态/UI 变化噪声）产生 2 个假界 → 收严为**断档需 ≥2 样本或识别到 0**（`test_momentary_gap_same_gens_not_boundary` / `test_sustained_gap_same_gens_is_boundary` 固化）。真界 frame_10_40 前是 7 样本持续 None，跨头像 NCC≈−0.03，与局内噪声明显分离。

**真实帧集成**（`tests/test_prescan.py`，共 11 项 PASS）：BV1Uu8z6eEVM 目录（110 帧@10s）→ 锚点 (120,847) ratio 0.91、恰 1 个边界在 640s(frame_10_40)；BV16QtT6ZEPq（单局）→ 0 边界。

**检测器/产线打通**：
- `StreamingDetector` 新参数：`anchor_prior`（预扫锚点，跳过 WAIT 直入 CALIBRATE）、`detect_match_end`（默认 True；预扫判单局时置 False 以彻底避免假切，BV1aat 残留 ~290s 假界即由此压掉）、`force_new_match`（RECORD 态强制开新局，供预扫边界使用；抽取 `_start_new_match` 复用换局分支）。新测试 `TestAnchorPrior` 3 项 + `TestPrescanDrivenEnd` 2 项（全过）。
- `run_pipeline.py` 新增 `run_video_prescan`：预扫 → `plan_prescan`（anchor_ok / single / boundaries / segments）→ 落盘 `report/{bvid}/prescan.json` 预检报告 → 带 `anchor_prior` 全量跑；单局 `detect_match_end=False`，多局在预扫边界处 `force_new_match`（段内 RECORD 0/None 换局仍兜底，多出段照常写回）；锚点不可信时回退旧 WAIT 产线 `run_video`。CLI 加 `--prescan` / `--prescan-interval`。新测试 `tests/test_pipeline_prescan.py` 8 项 PASS（plan/报告/锚点回退/强制切段均 mock 验证）。

**验证**：`tests/test_prescan.py` 11 PASS、`tests/test_pipeline_prescan.py` 8 PASS、`test_stream_detector.py` 相关新类 PASS；此后 2026-09-07 已跑全量回归 **84 tests PASS**（~583s，覆盖 §3.14 新增与既有路径，见注意事项）。

**注意事项 / 需要留意的点**：
1. ✅ **全量回归已跑**（2026-09-07，base python `python -m unittest discover -s tests`）：**84 tests PASS**（~583s，含 §3.14 新增 `TestAnchorPrior`/`TestPrescanDrivenEnd`/`test_pipeline_prescan` 及既有路径），旧路径未受影响。
2. ✅ BV1aat 单局 dataset 合并（4 行 → 1 行，截 ~405.5s，label=-1）已完成（2026-09-07，离线拼接 match1~4 features，796 帧；未重跑 BV1aat）。
3. prescan 阈值现为硬编码常量（`PORTRAIT_NCC_THR 0.8`=用户定、`PORTRAIT_MIN_STD 8`、`MIN_ANCHOR_SCALE 0.9`、portrait 有效对 ≥2、`ANCHOR_MIN_RATIO 0.5`），样本多了再调。
4. 同值+单样本断档不再视为候选 → 极少见「5→5 且两局间仅 1 不可读帧」会被预扫漏切，交给段内 RECORD 兜底。
5. `asset/icon_executed.png`（2026-09-06 生成，代码中无引用）未加入提交，如需 executed 状态（§6 待办 7）再纳入。
6. §3.13 所述 `run_video` 返回 `closed=[]` 的统计口径缺陷仍在（本提交未改）；`run_video_prescan` 按 `_encode_match` 返回值累计，其 records 数可靠。
7. 旧 `run_video`/`run_frames_dir` 路径与默认参数行为完全不变，既有测试不受影响（设计如此，留给全量回归确认）。

### 3.15 全量 UT 提速 ~10x（583s → 56s，2026-09-07）

**目标**：全量 unittest 从 ~583s 压到 ~1min 内。实测最终 `python -m unittest discover -s tests` = **84 tests / ~56s / OK**（优化前 ~583s）。

**热点根因**：`hud_anchor.find_gen_anchors` 对每帧全图跑 17 尺度 `matchTemplate`（1080p 一帧 ~1.9s），被 prescan 每 10s 采样、WAIT/每帧 detect_anchor 大量调用；测试反复 `imread` 同一批真实帧。

**改动（均为语义等价的性能优化，全量回归为正确性门）**：
- `hud_anchor.py`：
  - `detect_anchor(prior=...)` 先验搜索限制在锚点附近 ROI（`_prior_roi` ±96px），不再全图扫 17 尺度（RECORD 每帧路径主获益）。
  - no-prior 大帧（≥`FAST_SEARCH_MIN_PIXELS`≈1080p）走 `_fast_full_search`：降采样 ds=0.4 粗扫 → 按位置聚类 → 原生分辨率局部精化（pad 24）。只保留 scale≥0.9 候选（WAIT/预扫下游本就丢弃小尺度菜单噪声）。小帧（720p test1）保持原全量路径，结果逐位一致。
- `calibrator.calibrate_hook_slots`：同一文件原在 4 个 hook 区域循环里被读 4 次，改为每文件只读一次（`frames` 可选入参可传已读内存帧，完全跳过读盘）。
- `make_report.pick_opening_frame`：新增可选 `frames` 预加载映射避免重复 `imread`。
- `stream_detector._finalize_calibration`：校准帧本在内存（`_calib`），传内存帧给上述两函数，省掉写盘后读回的往返。
- `tests/test_stream_detector.py`：`_sorted_frames()` 模块级缓存（真实帧在各测试间只读不 mutate）。

**正确性验证**：ds=0.25/0.35 曾致漏检回退；最终 ds=0.4 全量真实帧回归（prescan BV1Uu/BV16 锚点/边界、stream WAIT/换局/槽位、hook/gens 真值）全部通过，84 tests OK。

### 3.16 新视频 BV1QUt766Etg 验证 + 鲁棒性修复 + 并行产线（2026-09-07）

**背景**：用新视频 BV1QUt766Etg（1080p30，21.4min，641.8MB，`picture/raw_videos/` gitignore）验证 10s 预扫与检测，暴露一系列 bug 并修复；随后实现 match 级多进程产线提速。

**Bug 修复（TDD 各带回归测试）**：
- `gens_counter` 越界/空 crop 崩溃：伪锚点使 `gens_row` 相对坐标算出负 x → 空 crop → `cvtColor` 崩。加 `_valid_crop` 守卫（越界返回 None），`count_gens`/`GensTracker.update` 均判无 HUD。`test_gens_counter.TestGensCounterOutOfFrame`。
- 同源问题也出现在状态识别路径：`make_report.classify`（空 crop→unknown）、`build_opening_refs`（返回空 healthy）、`pick_opening_frame`（跳过越界 crop 帧）。
- WAIT 可能锁到"区域内稳定但非 HUD"的伪锚点（本片转场段 `(458,797)@1.3` 稳定多帧被锁）→ CALIBRATE 建在非 HUD 帧上，第二局全程 unknown/gens 0。修复：`StreamingDetector` 新增 `_anchor_plausible`（锚点须使 4 头像+gens 区域落在帧内）与**确认式 WAIT `_wait_prior`**（给 anchor_prior 时先确认先验附近确有 HUD 图标 ≥wait_min_frames 帧才开局）。`test_waits_until_hud_confirmed_before_match_start` 等。

**prescan 改进（针对本片两类问题）**：
- **锚点共识改为"全候选跨帧"**：原逐样本取单帧最高分，被常驻高分伪匹配（本片左缘 x≈10 scale1.1，score 0.74–0.80）压过真实 HUD → anchor_ratio 0.40/不置信。`_consensus_candidates` 按不同帧支持数选主簇 → 本片 **0.403→0.705，anchor_ok=true**。
- **头像骤变切局规则**：残局 gens 恒 0/None 不再回 5 时旧 gens 规则漏判/滞后（边界被推到 950s，真值 ~630s）。新增"连续 ≥2 样本 4 头像 NCC 骤降 = 旧局结束"边界，有头像骤变时优先于 gens 回 5（避免残局后期伪 5 多切）。本片边界 **950→630**。`prescan._avatar_boundaries` + `TestAvatarBoundary`。

**并行产线（`run_pipeline.run_video_parallel`，CLI `--prescan --parallel`）**：
- 预扫分段后，**每个对局一个独立进程**从本局起始检测；第二局起在转场后用真实 HUD 重建基线（顺带修复整片 force 切局把基线建在转场帧的问题）。
- 解码改顺序读取按间隔取样（`_iter_window_frames`，免每帧 seek ~30x）。
- 落盘后按 `min_frames`（默认 30）过滤过短局，避免 montage 抖动产生的 4 帧垃圾段入 dataset。
- 单局/锚点不可信自动回退 `run_video_prescan` 原路径。
- 实测 BV1QUt766Etg 全流程 **255s ≈ 4min15s**（预扫 ~50s + 两段并行 ~3.5min），相比旧估算 60–90min 大幅提速。

**BV1QUt766Etg 检测结果（dataset 现有 2 行 label=-1，保留待后续标注）**：
- match_1 11.0–629.5s（1243 帧）：第一局，健康/受伤演化，gens 5→4→2→1→0。
- match_2 675.0–1286.5s（1224 帧）：第二局，内部 montage 转场不再错误切成多个局；状态可读，gens/头像可持续输出。

### 3.17 注意点 / 接手提示（2026-09-08，显式列出）

1. **运行环境命令**：全量 UT = `python -m unittest discover -s tests`（base Python 3.8.18，现 101 tests ~53s）。多局视频全量 = `python run_pipeline.py <mp4> <bvid> --prescan --parallel`（`run_video_parallel`；单局/锚点不可信自动回退旧 `run_video_prescan`）。强制按单局处理 = `python run_pipeline.py <mp4> <bvid> --single-match [--end-at SECONDS] [--anchor X Y SCALE]`（锚点默认自动识别，`--anchor` 仅作可选覆盖）。新视频下载用 dbd env `python -m yt_dlp -f 30080 --write-info-json -c -o "picture/raw_videos/%(id)s.%(ext)s" <url>`（bilibili 需 `--add-header "Referer:https://www.bilibili.com/"` + Chrome UA 防 HTTP 412）。
2. **BV1QUt766Etg 是剪辑向视频**：用户确认真实为两局；当前已按预扫边界 630s 合并为两条记录，仍建议只作检测质量验证，不直接作为训练样本（与 BV1aat 结论一致，训练样本仍应以完整单局视频为准）。
3. **`run_video_parallel` 的帧不落盘**：worker 把帧写进临时 scratch 后删除（只保留 CSV 入 report + dataset）。如需可视核对帧需另行抽取或改代码（frames_root 参数目前只影响单局回退路径）。
4. **确认式 WAIT 语义变化**：给 `anchor_prior` 后不再立即开局，须先验附近连续出现 HUD 图标（`wait_min_frames` 帧）才 CALIBRATE——这修掉了"转场帧建基线→整局 unknown"的 bug，但也意味着**开局前无 HUD 的帧不会产生行**（属预期）。
5. **prescan 头像骤变规则优先级**：当存在"连续 ≥2 样本 4 头像 NCC 骤降"时会丢弃该窗口内 gens 回 5 的伪边界（否则如 BV1QUt 会把第二局开头误切到 950s）。若未来出现"换人不伴随头像骤变"的多局视频，此规则可能漏切——届时需再评估（现由 RECORD 兜底）。
6. **`run_video_prescan` 旧路径的换局**仍用整片单检测器 + force_new_match，对"第二局开局在转场后"的基线处理不如并行路径可靠；新视频优先走 `--parallel`。
7. **左缘伪锚点(scale≈1.1, x≈10) 广泛存在**（BV1QUt 每帧都可能命中，score 0.74–0.80）：`_anchor_plausible`/确认式 WAIT 已拦截；但 `find_gen_anchors` 本身仍会返回它，任何直接调它的新代码需自己过滤。
8. `executed≈dead` 已完成：运行时报告状态保留为 `executed`，`dataset_encoder` 编码时归一为 `dead`，因此 dataset 仍保持 30 维；素材实际为 `asset/icon_executed.jpg`（旧文档中的 `.png` 已按实际文件修正）。
9. 调试 montage PNG（`picture/BV1QUt766Etg_*.png`）、`report/BV1QUt766Etg/` 为本地验证产物，不入库。

### 3.18 BV1pht96fEjN 全片端到端与 executed 实测（2026-09-08）

- 使用命令：`python run_pipeline.py picture/raw_videos/BV1pht96fEjN.mp4 BV1pht96fEjN --prescan --parallel`。
- 预扫结果：`anchor_ratio=0.453 < 0.5`、`anchor_ok=false`、边界 `590s`；按设计自动回退旧版 `run_video`，本次**没有实际走 match 级并行**。
- 首次自动跑耗时约 **95s**，错误地产出 4 个 CSV；该结果已按用户真值废弃，不能作为正式分局结论：
- 该次自动分段的 match1~4 详情已废弃，不作为当前数据真值；真实单局结果记录在 §3.19。
- 已修：`run_video` 不再从已被 encoder 消费的 `closed_q` 或 dataset 总行数统计；现在返回本次实际 `closed/matches/records`。
- **注意**：BV1pht 的 prescan anchor 不足 0.5，导致本次未享受 parallel 路径；若要验证并行性能或清理短段，应先审阅 `prescan.json`/match1，再决定是否降低 anchor 阈值或增加人工 anchor 入口。

### 3.19 BV1pht 真值纠正（2026-09-08）

- 用户确认：`BV1pht96fEjN` 实际是**单局**，于 `14:10.5 = 850.5s` 结束；此前按 prescan/旧换局规则得到的 4 段不是 4 局，判断错误。
- 已用固定真值隔离重跑：anchor=`(141,804) scale=1.6`、`detect_match_end=False`、窗口 `[0,850.5s)`；产出单 CSV **1690 帧**，时间 `5.5–850.0s`，gens 主走势 `5→4→3→2→0`，p2/p3 大量 `executed` 命中。
- 已替换正式产物：删除旧 `report/.../match_2~4`，保留单一 `report/.../match_1/detect_report.csv`；`dataset/videos.jsonl` 中 BV1pht 旧 4 行替换为单行 `match=1`、1690 features、label=-1。
- **注意**：该次修正当时用了人工锚点/结束时间；现已改为**自动识别**（见 §3.21），无需人工锚点即可得到单局结论。

### 3.21 自动锚点/切局的可靠性修复（2026-09-08）

**背景**：用户指出"人工确认锚点不靠谱，还是要自动识别"。排查 BV1pht 自动失败根因后修复，四个真值视频现已全部自动判定正确。

- **根因 A（锚点阈值过高）**：`find_gen_anchors` 默认 `min_score=0.70`，而 BV1pht 发电机图标 NCC 实测仅 `0.66~0.69`（与 §3.8-A 一致）。抽样验证：真实图标命中率 `min_score=0.70 → 35%`、`0.55 → 95%`；共识支持数被严重低估，`anchor_ratio` 误为 `0.453 < 0.5` → `anchor_ok=false` → 回退旧产线并被误切。
  - 修复：`prescan.run_prescan` 新增 `anchor_min_score`（默认 `ANCHOR_MIN_SCORE=0.55`，与 gens 图标识别 `GEN_ICON_THR` 一致）；跨帧共识本身能滤除不稳定伪匹配，降低阈值安全。
  - 效果：BV1pht `anchor_ratio 0.453 → 0.942`，`anchor_ok=true`。
- **根因 B（gens 同值+断档切局过松）**：`find_boundaries` 的"同值+≥2 帧断档复现"分支用 `portrait_thr=0.8`，而 BV1pht 530~580s 只是 gens 暂时读不到（HUD 仍在，sim≈0.78）却被判换局（原边界 590s）。
  - 修复：该分支单独用更严阈值 `SAME_VALUE_PORTRAIT_THR=0.5`（`g > last_g` 仍用 0.8）。
- **根因 C（头像骤变规则把局内波动当换局）**：降阈值后 BV1pht 出现假边界 210s——210/220s 连续 sim 0.22/0.31 只是局内头像状态变化，230s 又恢复。
  - 修复：`_avatar_boundaries` 增加"确认"——骤变段结束后头像须与段前**持续不同**（`portrait_similarity(post, pre) < AVATAR_CONFIRM_THR=0.5`）才算换局；恢复相似则判为局内波动不切。
- **四视频自动回归**（无人工锚点）：

  | 视频 | anchor_ratio | anchor_ok | single | 边界 |
  |---|---|---|---|---|
  | BV1Uu8z6eEVM | 0.836 | true | false | 640 |
  | BV16QtT6ZEPq | 0.700 | true | true | — |
  | BV1pht96fEjN | 0.942 | true | **true** | — |
  | BV1QUt766Etg | 0.798 | true | false | 630 |

- **BV1pht 自动端到端验证**（隔离临时目录，未用人工锚点）：`run_video_parallel` 自动走单局分支 → 1 个 match、1707 帧、`5.5–858.5s`、约 4min45s；与真值（单局）一致（尾部含结算，可不截或加 `--end-at`）。
- **`--single-match` 语义调整**：不再要求人工 `--anchor`；锚点默认自动识别，`--anchor` 改为可选覆盖，`--end-at` 可选（默认到视频结束），用于强制单局/截断。
- 全量回归：**101 tests PASS，约 53s**。

### 3.20 BV1QUt 流程耗时 profile 与提速（2026-09-08）

- 在临时目录对 BV1QUt 做基线/优化后对比，正式 dataset/report 未被改动：
  - 基线：预扫 `58.0s`，并行检测 `224.3s`，总 `282.3s`。
  - 优化后：预扫 `38.5–43.5s`，并行检测 `192.0s`，总约 `230.5s`，提速约 **18%**。
- 优化点：
  - `prescan.run_prescan` 视频源采样帧缓存为 JPEG 内存数据，第二阶段不再对同一时间点再次 seek/decode；目录源仍直接复用帧对象。
  - `hud_anchor.detect_anchor` 带 `prior_scale` 时只扫描先验附近 5 个尺度（±0.2），不改变无先验全图搜索；600 帧实测检测从约 92ms/帧降至 71ms/帧。
- 全量回归：**94 tests PASS，约 53s**。
- **注意**：预扫 JPEG 缓存以降低内存峰值，但会比原始帧内存缓存慢约 5s；长视频应优先控制内存，不建议无限缓存原始 1080p 帧。
- 本次 profile 输出位于临时目录 `C:\Users\A\AppData\Local\Temp\opencode\BV1QUt_profile*`，未覆盖正式 dataset/report。

### 3.22 结局自动标注（label）与真值修正（2026-09-08）

**用户确认口径**：label = 该局**逃生人数 0–4**，看**最后一帧 HUD** 的 4 个幸存者头像图标，数 `escaped` 个数（`dead/executed/hooked/dying` 不算）。结算/计分画面不一定每局都有，不作为检测依据。

**实现**：
- 新增 `labeling.py`：`infer_label(states_per_frame, window=40)` 统计末尾窗口内出现过 `escaped` 的人数；全 unknown 返回 `None`。`infer_label_from_csv` 从 `detect_report.csv` 读取。
- `run_pipeline._encode_match` 接入：编码时自动推断 label，无法判定回退 -1。全过程无需人工标注。

**真值修正（原种子标签被证伪）**：
- 用户确认：BV1Uu 应为 **2**（原记 3，且原行实际跨了两局，需后续重生成）；BV16 应为 **3**（原记 1）。
- 用末段 HUD 自动推断验证一致：BV1pht **1**（p1 escaped，其余 executed，与用户一致）、BV1QUt 两局 **0/0**（用户确认）。
- 已更新 `dataset/videos.jsonl`：BV1Uu=2、BV16=3、BV1pht=1、BV1QUt=0/0；剪辑向 BV1aat 保持 -1（非训练样本）。
- **高亮逃出已修**（2026-09-08，借助图像目视确认）：逃跑图标是"穿门小人"，高亮/暗色**同形**。做法：从 BV16 `04_50` p1 取高亮图标，**去背景 + 紧裁**存为 `asset/icon_escape_highlighted.png`；`make_report.classify` 里作为**独立高阈值补充**（`load_escape_highlighted` / `ESCAPE_HIGHLIGHTED_THR=0.85`），不混入通用模板。量化：BV16 高亮 p1=0.921（采纳）；BV1Uu 普通头像误报仅 ≤0.694（被 0.85 挡掉）。效果：BV16 自动 label 2→**3**（=真值），BV1Uu/BV1pht/BV1QUt 无新增误报。
- **BV1Uu 重生成**：原单行（110 帧，混了 0–640s 与 640–1090s 两局）已重分段为两局：match_1（62 帧，0–630s）、match_2（46 帧，640–1090s）。match_2 按用户真值标 `label=2`；match_1 无真值标 `-1`。
- **注意**：BV1Uu 为 10s 稀疏帧，重分段后两局自动推断均为 0（帧内从未识别到 escaped，目视 18:10 也无逃跑图标），与 match_2 用户真值 2 冲突——该旧帧目录**未覆盖到真正逃生时刻**（或用户记忆偏差）。需后续用源视频 0.5s 重抽才可靠（当前无 BV1Uu mp4）。
- 全量回归：**110 tests PASS，约 54s**。

### 3.23 videos.jsonl 存储优化：紧凑帧元组（2026-09-08）

- **动机**：每帧原存 30 个浮点 one-hot（大量 `0.0/1.0`），154 B/帧；`videos.jsonl` 785 KB/5091 帧，上千局将膨胀到数百 MB。
- **方案（用户选定）**：行键 `features`（30 维）→ `frames`（每帧 10 个整数）
  `[s1,s2,s3,s4, h1,h2,h3,h4, gens, t_half]`：状态用类别索引（unknown→0、executed→4）、`t_half` 复用 `parse_time` 半秒整数。
- **实现**：`dataset_encoder.compact_frame`（编码）、`frames_to_features`（还原 30 维，供模型/兼容）；`encode_csv` 输出 `frames`。因下游序列模型尚未实现，此格式变更成本最低。
- **效果**：**785 KB → 176 KB（约 4.5x）**，仍为可读 JSONL。
- 文档 `docs/dataset_format.md` 已同步；旧 `features` 键废弃。全量回归 **113 tests PASS**。

## 4. 测试数据与真值

- 示例帧：`picture/test1/`，12 帧 1280×720（frame_0000~0011），0/10/11 无发电机图标（0=开局、10/11=修完）
- BV1 整局：`picture/BV1Uu8z6eEVM/` 110 帧（1080p），BV16 整局：`picture/BV16QtT6ZEPq/` 30 帧（1080p），均由 `extract_frames.py` 抽帧
- 生成物：`picture/crops/frame_XXXX/survivor_pN.jpg`（头像）、`picture/portrait_montage.png`（48 头像蒙太奇，用户核对了真值）、`picture/hook_montage.png`（48 hook 蒙太奇）
- **gens 数字真值**（GensTracker 测试用，`tests/test_gens_tracker.py`）：
  - BV1 第 1 局：5(00_00~01_20)→4(01_30~03_20)→3(03_30~05_30)→2(05_40~09_00)→1(09_10~09_30)→0(09_40~10_30)；frame_10_40 换局回 5
  - BV16：5(00_10~01_30)→4(01_40)→3(02_00)→2(02_20~03_30)→1(03_40~03_50)→0(04_00+)；01_50/02_10/03_00 为过渡帧
  - test1：5(0001)→4(0002~0005)→3(0006)→2(0007~0009)→0(0010/0011)；0000=None
- **头像状态真值**（经用户确认）：
  - frame_0000：全 unknown（无 HUD）
  - frame_0001：全 healthy
  - frame_0002：p4 injured（其余 healthy）
  - frame_0003：p3 hooked，p4 dying
  - frame_0004：p3 hooked
  - frame_0006：p2 hooked
  - frame_0007：p2 injured，p4 hooked
  - frame_0008：p4 injured
  - frame_0009：p4 dead
  - frame_0010：p4 dead，p2 hooked
  - frame_0011：p3 escaped，p2/p4 dead
- **hook 计数真值**（48 格全对）：

  ```
           p1 p2 p3 p4
  frame0    0  0  0  0
  frame1    0  0  0  0
  frame2    0  0  0  1
  frame3    0  0  1  1
  frame4    0  0  2  1
  frame5    0  0  2  1
  frame6    0  1  2  1
  frame7    0  2  2  2
  frame8    0  2  2  2
  frame9    0  2  2  2
  frame10   0  2  2  2
  frame11   0  2  2  2
  ```

## 5. 已知问题 / 边界情况

- 对话助手可读图但**代价高**：仅在必要时看少量图；大量图像分析仍以代码数值 + 用户目视为主
- **视频开头/局间为菜单时，流式检测器必须跨帧确认锚点**（WAIT 用滑动窗口共识，见 3.7）；单帧 max-score 会被菜单小尺度误报抢占
- 锚点曾在 frame_0011 被假匹配（scale=1.3 @ (344,567)）污染配置，已用 `fix_anchor.py` 恢复并加 `TRUST_SCORE` 防护
- 健康基线依赖开局全健康帧；若某局开局即有异常需重新审视
- 深伤口（Deep Wound）状态尚未实现
- hook 槽位不与锚点等比缩放，每视频必须用 `calibrate_hook_slots` 校准（见 3.3）；若某局视频帧数过少（<5）槽位校准可能不稳
- gens 数字识别模板库 `asset/gens_digits/` 已含 BV1+BV16 样本（45 张），1/2 样本较多、3/4/5 较少；新视频遇到匹配不佳的数字时追加样本迭代（见 3.4）
- 过渡帧（数字变化瞬间，如 BV16 01_50 图标匹配 0.68<0.70）可能判为 0，10s 间隔采样下可接受

## 6. 待办（下一步）

**main（当前分支）**：Task1-4 + Task5 修复 + UT 提速(§3.15) + BV1QUt 修复与并行产线(§3.16) + BV1pht 真值纠正(§3.19) + 自动锚点/切局(§3.21) + 结局自动标注(§3.22) + 存储优化(§3.23) 均已推送；另 5 支视频已 **0.5s 密集重建**（§3.5，本会话，待提交）。dataset 现 7 行（BV1Uu 2 局、BV16=3、BV1pht=1、BV1QUt=0/0；BV1aat=-1、BV1Uu m1=-1）。原 113 tests ~57s PASS；现 **130 tests PASS**（§3.5）。

**已完成（本会话）**：
- ✅ 全量 UT 提速 583s→~57s（§3.15）。
- ✅ BV1aat 单局 dataset 合并（§3.14/§3.15）。
- ✅ BV1QUt766Etg 验证：预扫崩溃修复、锚点全候选共识（0.40→0.71）、头像骤变切局（950→630）、确认式 WAIT、`run_video_parallel` 按局多进程 + min 帧过滤（§3.16）。
- ✅ executed≈dead 状态（§3.16 后）。
- ✅ BV1pht 真值纠正为单局 + 自动锚点/切局可靠性修复（§3.18/§3.19/§3.21，四视频自动回归全对）。
- ✅ 结局自动标注 `labeling.py`（末段 HUD escaped 计数）+ 种子标签修正 + 高亮逃出识别（§3.22）。
- ✅ `run_video` 统计口径修复 + `--single-match` 自动锚点（§3.19/§3.21）。
- ✅ `videos.jsonl` 紧凑帧元组存储（785KB→176KB，§3.23）。

**下一步（按优先级）**：
1. ✅ 对局序列数据管道：`match_dataset.py` → `match_model.py` → `train_sequence.py` → `predict_live.py` 已实现（见 §3.5「实现进展」；环境 torch 2.2.2+cpu 可用；全量 130 tests PASS）。**当前仅流程 smoke test**。
2. ⏳ `gate_ui` 大门状态识别 + 真实帧回归。
3. ⏳ BV1Uu 源视频 0.5s 重抽并重标（当前无 mp4；10s 稀疏帧无法自动标注 match_2）。
4. 数据积累：更多完整单局视频 → 自动检测 + 自动标注，扩充训练集（上千局）→ 之后才有意义的模型训练/评估（按局切分 + 混淆矩阵/MAE）。
5. 模型超参调优（hidden 层数/dropout/lr）+ 整局胜率/结局概率走势图输出。
6. （可选）局内多分片并行（warm-up 状态交接）进一步提速。

## 7. 环境说明

- Windows；dbd conda env：**Python 3.12.14**、torch 2.2.2+cpu、numpy 1.26.4、cv2 4.8.0、matplotlib 3.11.1、Pillow 12.3.0（见 `requirements.txt`）
- 全量测试：`& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`（现 130 tests）
- 训练：`python train_sequence.py --videos dataset/videos.jsonl --out models/match_gru.pt --epochs 30`
- 推理：`python predict_live.py models/match_gru.pt <detect_report.csv>`

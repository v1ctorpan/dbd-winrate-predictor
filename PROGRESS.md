# DBD 胜率预测项目 — 进展与设计文档

> 最后更新：2026-09-03
> 状态：HUD 区域校准完成；头像状态识别、hook 计数、发电机剩余数识别均已对测试数据 100% 正确；发电机数字识别已升级为**通用模板库 + 时序状态机（GensTracker）**，无需 per-video 硬编码；**多线程数据产线（下载→抽帧→检测→编码）Task1-4 已在 main 全部完成（41 测试全绿）**；当前在 `data_pipeline` 分支执行 Task5「真实视频端到端跑通」，期间**修复了流式检测器 WAIT 状态被菜单误报抢先触发、锁定错误锚点**的 bug，小样本已验证修复生效。

## 分支与提交状态（2026-09-03）

- `main`：Task1-4 已提交（`80bdf87` 半秒帧命名/parse_time → `f3de624` 流式检测器 → `8f6f730` dataset_encoder append/read → `83deb92` 三线程 run_pipeline），全量 41 测试通过（含本次修复带来的 +1）。**相对 origin/main 领先 3 个提交未推送**。
- `data_pipeline`（当前分支）：从 main 的 `83deb92` 切出，专做 Task5「BV1pht96fEjN 真实视频端到端跑通」。本分支新增：
  - `stream_detector.py`：修复 WAIT_ANCHOR 单帧 max-score 误判（见 3.7）。
  - `tests/test_stream_detector.py`：新增 `test_wait_requires_stable_position_across_frames`（菜单抖动不触发、稳定同位置触发）。
- 产物：BV1pht96fEjN.mp4（1080p，14.3 分钟，435 MB）已下载到 `picture/raw_videos/`（gitignore）；锚点经探查 + 用户目视确认 = `(142,806) scale=1.5`（与 BV1 的 (121,847)@1.3 不同，hook_regions.json 仅 key 到 BV1Uu8z6eEVM，不会误用）。
- **待办见 §6**。

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

### 3.5 对局序列数据管道与结局预测模型（设计完成）

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

### 3.6 多线程数据产线（设计定稿，实现进行中）

为把"下载→抽帧→检测→编码"串成自动化产线并支持多线程提速，设计已定稿：`docs/superpowers/specs/2026-09-02-pipeline-multithread-design.md`，实现计划 `docs/superpowers/plans/2026-09-02-pipeline-multithread.md`（5 任务 TDD，直接提交 main 无分支）。

- 抽帧时间精度升级：`parse_time` 支持**半秒精度**，帧名 `frame_MM_SS.0.jpg`（整数秒）/ `frame_MM_SS.5.jpg`（半秒）；`extract_frames.py` 的 `frame_name(t)` 与 `--interval 0.5` 兼容（commit `80bdf87`，全量 34 测试 PASS）。
- 检测侧复用既有函数：`make_report.pick_opening_frame/build_refs/classify/build_opening_refs`、`calibrator.calibrate_hook_slots`（路径版）、`gens_counter.GensTracker`、`hook_counter.count_all`；新增流式检测器状态机 WAIT_ANCHOR→CALIBRATE(budget=12)→RECORD，`apply_hook_cfg` 扩展支持 hook_names 列表。
- 编码侧：一局 = 一行 JSONL 追加进 `dataset/videos.jsonl`，`id="{BVid}:{match_no}"`，label=-1 待标注。
- 任务状态：Task 1（帧命名 + parse_time）✅ commit `80bdf87`；Task 2（流式检测器）✅ commit `f3de624`；Task 3（追加式编码）✅ commit `8f6f730`；Task 4（三线程 run_pipeline）✅ commit `83deb92`；Task 5（BV1pht96fEjN 端到端）⏳ 进行中（当前 `data_pipeline` 分支）。

### 3.7 WAIT 锚点误触修复（data_pipeline 分支，2026-09-03）

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

- 画面对话助手无法直接查看图片（模型限制），一切图像分析依赖代码数值 + 用户目视确认
- **视频开头/局间为菜单时，流式检测器必须跨帧确认锚点**（WAIT 用滑动窗口共识，见 3.7）；单帧 max-score 会被菜单小尺度误报抢占
- 锚点曾在 frame_0011 被假匹配（scale=1.3 @ (344,567)）污染配置，已用 `fix_anchor.py` 恢复并加 `TRUST_SCORE` 防护
- 健康基线依赖开局全健康帧；若某局开局即有异常需重新审视
- 深伤口（Deep Wound）状态尚未实现
- hook 槽位不与锚点等比缩放，每视频必须用 `calibrate_hook_slots` 校准（见 3.3）；若某局视频帧数过少（<5）槽位校准可能不稳
- gens 数字识别模板库 `asset/gens_digits/` 已含 BV1+BV16 样本（45 张），1/2 样本较多、3/4/5 较少；新视频遇到匹配不佳的数字时追加样本迭代（见 3.4）
- 过渡帧（数字变化瞬间，如 BV16 01_50 图标匹配 0.68<0.70）可能判为 0，10s 间隔采样下可接受

## 6. 待办（下一步）

**data_pipeline 分支（当前焦点）**：
1. 深入排查 BV1pht96fEjN 样本 CSV 中 **gens 几乎全程 0** 与 **hooks 全程 0** 的可疑读数：
   - gens：`GensTracker` 开局 12 帧=5 后长期 0/None——需确认是"本局后续确实无图标（修完/大门已开）"还是区域/匹配在局内失效（锚点锁定 (142,806)@1.5 稳定，非菜单干扰）。
   - hooks：349 帧全程 0/0/0/0 即便出现 dying——需核对 `calibrate_hook_slots` 在新几何下槽位与 `count_hooks` 阈值/run 判定是否有效（hook 槽位已知**不与锚点等比缩放**，每视频必须单独校准，见 §3.3）。
2. 修复/确认后，对 BV1pht96fEjN 全片（14.3min）端到端跑 `run_pipeline.py`（预计 ~55min），产出 `picture/BV1pht96fEjN/match_N/`、`report/.../detect_report.csv`、`dataset/videos.jsonl` 的 `BV1pht96fEjN:N` 记录。
3. 校验数据集行（features 长度、id 带局号、label=-1）。
4. 更新 `docs/dataset_format.md` 与 `PROGRESS.md`，提交 data_pipeline 分支改动。

**main（Task1-4 已提交，待推送 + 产线收尾文档）**：
5. 推送 main 领先的 3 个提交（`f3de624`/`8f6f730`/`83deb92`）到 origin。
6. 将修复后的流式检测器改动（3.7）与 data_pipeline 分支成果合入/同步到 main 并推送。
7. 全量回归（41 测试）+ 最终文档核对。

**后续规划（已定稿未实现）**：
8. 实现对局序列数据管道（设计+计划已完成，见 3.5）：`dataset_encoder.py` → `match_dataset.py` → `match_model.py` → `train_sequence.py` → `predict_live.py`
9. 处理剩余 HUD 元素：
   - ~~发电机剩余数：`gens_row` 区域~~ ✅ `gens_counter.py`
   - 大门状态：`gate_ui` 区域
10. 结算画面自动标注结局（当前种子数据人工标注）
11. 数据积累：更多视频抽帧 → 编码 → 标注，扩充到上千局
12. 模型超参调优 + 胜率走势图输出

## 7. 环境说明

- Windows，Python 3.8.18（Anaconda base，注意不是 3.9+）
- cv2、numpy 1.22.3、matplotlib 3.5.1、PIL

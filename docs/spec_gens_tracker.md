# GensTracker 时序数字识别 — 设计规格

## 背景与目标

现状：`gens_counter.count_gens()` 是**无状态纯函数**，每帧独立做模板匹配。
它需要每视频手动指定参考帧（BV1/BV16 都硬编码在 `make_report.py`），无法用于用户后续的**实时间隔截图**场景（不可能提前看完整视频构建模板）。

目标：构建一个**状态化的 GensTracker**，只需一份**通用高清数字模板库**即可识别任意视频，并利用 gens 数字的**时序单调递减**特性提升鲁棒性。

## 已完成的 spike 验证

1. **BV1（1080p, scale=1.3）数字模板可跨视频通用**：
   - 识别 BV1：10/10 正确
   - 识别 BV16（放大到 scale=1.5）：5/5 正确
   - 识别 test1（缩小到 scale=1.0）：4/4 正确
   - 每个数字最高分与次高分有明显间隔。

2. **帧间全像素 NCC 可区分数字变化**（BV1 样本）：
   - 同数字：5→5 ncc=0.93、3→3 ncc=0.70、2→2 ncc=0.89
   - 不同数字：5→4 ncc=0.16、4→3 ncc=0.32、2→1 ncc=0.24
   - 但有重叠：4→4 ncc=0.51（偏低）、3→2 ncc=0.72（偏高，形状相近）

   → 单靠帧间 NCC 不可靠，必须**结合模板重识别 + 递减约束**。

## 架构设计

### 1. 通用高清模板库（asset/gens_digits/）

- 目录：`asset/gens_digits/`
- 文件：`digit_{n}_{seq}.png`（从 BV1 1080p 提取，scale=1.3，每数字 3-7 样本）
- 索引：`refs.json`：`{"5": ["digit_5_00.png", ...], "4": [...], "3": [...], "2": [...], "1": [...]}`
- 数字 0 不建模板（图标消失=0，见下）。
- 启动时加载一次，模板按当前帧数字框尺寸 resize 匹配。

### 2. GensTracker 类（gens_counter.py 新增）

```python
class GensTracker:
    def __init__(self, refs, gen=None, icon_thr=GEN_ICON_THR,
                 digit_thr=DIGIT_THR, track_ncc=0.85):
        self.refs = refs          # {digit: [img,...]} 高清模板
        self.gen = gen or 加载 gen.jpg
        self.icon_thr, self.digit_thr, self.track_ncc = ...
        self.prev_digit = None    # 上一帧识别结果 (1-5/0/None)
        self.prev_crop = None     # 上一帧数字框裁剪

    def reset(self):
        self.prev_digit = None
        self.prev_crop = None

    def update(self, frame, resolved, anchor) -> int|None:
        """处理单帧，返回 gens 数字。状态跨帧保留。"""
```

### 3. 识别流程（每帧）

```
1. 裁剪 gens_row 数字区 crop，灰度化
2. 若亮像素 < 20 → 返回 None（HUD 未渲染），并重置 prev_crop
3. 匹配 gen 图标 (matchTemplate)：
   - 图标匹配 < icon_thr(0.70) → 返回 0（图标消失=修完），重置 prev_crop
4. 取数字区左侧 DIGIT_X*scale 宽 digit_crop
5. 若 prev_crop 存在且 prev_digit 有效：
   计算 ncc = NCC(当前 digit_crop, prev_crop)  # resize 对齐
   - 若 ncc >= track_ncc(0.85)：沿用 prev_digit，更新 prev_crop，返回
6. 模板重识别：对每个数字的每个样本 resize 到 (dh,dw) 求 NCC，
   取最高分；best_score >= digit_thr(0.55) → best_digit
   - 若 best_digit > prev_digit：递减约束，沿用 prev_digit（除非 prev_digit 为 None）
7. 更新 prev_digit / prev_crop，返回结果
```

### 4. 与现有代码的关系

- 保留 `count_gens()`（无状态纯函数）供单帧测试和旧调用兼容。
- 新增 `GensTracker`，`make_report.py` 的 `process_video` 用 tracker 替代 count_gens。
- **不再需要** per-video 硬编码数字帧映射（删除 BV1/BV16 分支）。
- `_detect_match_swaps` / `_find_swap_start`：每个调用点**新建独立 tracker**（稀疏采样，跨局），或在每局循环内 reset。设计：在 process_video 主循环创建一个 tracker，在检测到换局（swap）时 `tracker.reset()`。

### 5. 换局处理

- 换局 = gens 从非 5 跳回 5（已由 `_detect_match_swaps` 检测）。
- 换局帧时 reset tracker（prev_digit/prev_crop 清空），避免用上局状态。
- 首帧（无 prev）自然走模板识别，无特殊处理。

### 6. 参数

| 参数 | 值 | 说明 |
|---|---|---|
| GEN_ICON_THR | 0.70 | 图标消失判定阈值 |
| DIGIT_THR | 0.55 | 模板识别最低置信 |
| LOW_THR | 0.45 | 低置信采纳阈值（best≥LOW_THR 且符合递减时仍采纳） |
| TRACK_NCC | 0.85 | 帧间沿用阈值 |
| DIGIT_X | 29 | 基础数字宽度（×scale） |

### 7. 模板库迭代（asset/gens_digits/）

- 模板库由**多视频样本累积**而成（当前 45 样本：BV1 + BV16 正确标注）。
- 新视频处理时先仅用现有库；识别结果人工核实后可**追加样本**到库（按数字分文件夹），逐步逼近通用。
- 迭代中若发现某视频渲染与库差异大（如 BV16 的 5 vs 3），追加该视频的样本即可显著提升匹配（实测 frame_01_30 的 5 从 0.59→1.0）。

## 验收标准

1. `GensTracker` 对 BV1/BV16/test1 全部帧的 gens 识别，与人工真值一致（BV1 110 帧、BV16 30 帧、test1 12 帧）。
2. 时序沿用生效：连续同数字帧走 step 5（ncc≥0.85 沿用），不依赖模板。
3. 递减约束生效：模拟"识别出比前帧大"场景，输出沿用前帧。
4. 现有 tests 全部通过（25/25），新增 tracker 测试。
5. 从 make_report.py 删除 per-video 硬编码 digit 映射。

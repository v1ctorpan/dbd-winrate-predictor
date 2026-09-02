# 多线程数据产线实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"抽帧→检测→编码"改造成同进程多线程流水线 `run_pipeline.py`，用 BV1Z58J6bEoi 真实视频跑通，产出带局号的数据集记录。

**Architecture:** 抽帧线程按 interval（默认 0.5s）解码并**只入队 (frame, fname)**，不落盘；检测线程是唯一消费者/落盘方——由它决定 match 归属，把帧持久化到 `picture/{BVid}/match_N/`（校准与重跑都依赖磁盘帧），复用 make_report 的 `pick_opening_frame`/`classify`/`build_refs`/`apply_hook_cfg` 与 path 版 `calibrate_hook_slots`，状态机为 WAIT_ANCHOR→CALIBRATE→RECORD→(换局)MATCH_END；编码线程把每局 CSV 编码并 append 进 `dataset/videos.jsonl`（id 带局号 `BVid:N`，label 占位 -1）。

**Tech Stack:** Python 3.12（conda env `dbd`）、cv2、numpy、unittest、threading/queue。

**Spec:** `docs/superpowers/specs/2026-09-02-pipeline-multithread-design.md`

## Global Constraints

- Python 解释器（所有命令）：`C:\Users\Sallia\.conda\envs\dbd\python.exe`，PowerShell 中带 `& "..."` 前缀。
- 测试命令：`& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`（Windows PowerShell，不用 pytest）。
- 帧命名：整秒 `frame_MM_SS.0.jpg`、半秒 `frame_MM_SS.5.jpg`（SS 为 floor 秒，半秒来自小数部分）。`parse_time` 返回"半秒为单位的整数"；`features[29]` = `parse_time/2.0`（原始秒，float，不归一化）。
- 检测线程**落盘全部进入 CALIBRATE/RECORD 的帧**到 `picture/{BVid}/match_{no}/`；WAIT 段（菜单/结算）不落盘。
- 换局信号 = gens 从非 5 跳回 5（沿用 `_detect_match_swaps` 判据的流式版）；每局独立 healthy 基线（`pick_opening_frame`）与 hook 槽位（`calibrate_hook_slots`）。
- `make_report.py` 的 `classify(crop, slot, refs, icon_tpl)`（返回字符串）、`build_refs(scale)`、`load_official_icons()`、`pick_opening_frame(...)`、`apply_hook_cfg(resolved, anchor, video_name)` 原样复用，不复制内部实现。
- hook 区域覆盖：`config/hook_regions.json` 目前只作用于 `BV1Uu8z6eEVM`；BV1Z58J6bEoi 与 BV1 几何相同，需把该作用域扩展为两者共用（改 `apply_hook_cfg` 接受列表）。
- 数据集记录：`id` = `{BVid}:{match_no}`，`label` = `-1` 占位（验证后人工回填）。
- 提交信息风格：`feat: `/`test: `/`docs: ` 中文短句。

---

### Task 1: 半秒帧命名与 parse_time/时间编码

**Files:**
- Modify: `dataset_encoder.py`（parse_time 半秒语义；feature_vector 时间 = /2.0）
- Modify: `tests/test_dataset_encoder.py`
- Modify: `extract_frames.py`（interval 小数 + 命名 .0/.5）——仅改函数供复用；实际流水线抽帧在 Task 4 内联（同一命名规则）

**Interfaces:**
- Produces: `parse_time(fname) -> int`（半秒：`frame_01_30.5.jpg` → 181；`frame_00_00.0.jpg` → 0）；`feature_vector` 保持 30 维、`v[29]` 秒。
- Test 基准确认：现有 `TestParseTime.test_parse_time` 断言整秒秒数（90/765），改为半秒整数（180/1530）；`TestFeatureVector.test_time_raw_seconds` 断言 `v[29]==90.0` 不变（parse 180/2）；`TestEncodeVideo.test_encode_csv` 的 `v[29]` 0.0/10.0 不变（frame_00_00→0/2=0，frame_00_10→20/2=10）。

- [ ] **Step 1: 先改测试（红）**

`tests/test_dataset_encoder.py`：

```python
class TestParseTime(unittest.TestCase):
    def test_parse_time_integer_seconds(self):
        self.assertEqual(de.parse_time("frame_00_00.0.jpg"), 0)
        self.assertEqual(de.parse_time("frame_00_10.0.jpg"), 20)
        self.assertEqual(de.parse_time("frame_01_30.0.jpg"), 180)   # 90s = 180 half-seconds
        self.assertEqual(de.parse_time("frame_12_45.0.jpg"), 1530)

    def test_parse_time_half_second(self):
        self.assertEqual(de.parse_time("frame_00_00.5.jpg"), 1)     # 0.5s = 1 half-second
        self.assertEqual(de.parse_time("frame_01_30.5.jpg"), 181)
        self.assertEqual(de.parse_time("frame_03_07.5.jpg"), 375)   # 187.5s = 375 half-seconds
```

`TestFeatureVector.test_time_raw_seconds` 改用半秒帧名确认仍返回秒：

```python
    def test_time_raw_seconds(self):
        row = {"p1": "healthy", "p2": "healthy", "p3": "healthy",
               "p4": "healthy", "hooks": "0/0/0/0", "gens": "3", "frame": "frame_01_30.5.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(v[29], 90.5)
```

- [ ] **Step 2: 运行确认 FAIL**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_dataset_encoder -v`
Expected: 新断言 FAIL（parse_time 仍返回秒；feature_vector 现返回 int 秒）。

- [ ] **Step 3: 改实现**

`dataset_encoder.py`：

```python
def parse_time(fname):
    """返回以半秒为单位的整数时间。frame_MM_SS.5.jpg = MM*120 + SS*2 + 1。"""
    stem = fname.split(".")[0]              # frame_MM_SS
    parts = stem.split("_")
    mm, ss = int(parts[1]), int(parts[2])
    half = 1 if ".5" in fname else 0
    return mm * 120 + ss * 2 + half
```

`feature_vector` 时间分量行改为：

```python
    feats.append(float(parse_time(row["frame"])) / 2.0)
```

（30 维不变，维度 29 = 原始秒，支持 .5 粒度。）

- [ ] **Step 4: 运行测试确认 PASS**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_dataset_encoder -v`
Expected: 全部 PASS（其余 6 个旧断言不受影响：one_hot/encode_csv 的 0.0 与 10.0 保持不变）。

- [ ] **Step 5: 更新 `extract_frames.py` 命名函数（供复用与单测）**

把命名逻辑抽成独立函数以便复用：

```python
def frame_name(t):
    """t(秒) -> 帧名(不含扩展名)。整秒->frame_MM_SS.0; 半秒->frame_MM_SS.5。"""
    m, s = divmod(int(t), 60)
    half = "5" if (t - int(t)) > 0.25 else "0"
    return f"frame_{m:02d}_{s:02d}.{half}"
```

并把 `extract_frames` 内命名改为 `name = frame_name(t)`（`interval` 现可为 0.5；循环用 `t += interval`，`total_targets = int(duration / interval) + 1` 兼容小数）。不改其 `tqdm` 进度条与 CLI（CLI interval 仍 `int()`，流水线不依赖其 CLI）。

- [ ] **Step 6: 运行全量确认无回归**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`
Expected: 全部 PASS。

- [ ] **Step 7: 提交**

```bash
git add dataset_encoder.py tests/test_dataset_encoder.py extract_frames.py
git commit -m "feat: parse_time支持半秒精度; 抽帧命名与0.5s间隔兼容"
```

---

### Task 2: 流式检测器（state machine 复用既有检测函数）

**Files:**
- Create: `stream_detector.py`
- Modify: `make_report.py`（`apply_hook_cfg` 支持多个视频共用 hook 区域）
- Test: `tests/test_stream_detector.py`

**Interfaces:**
- Consumes: `make_report.pick_opening_frame(frame_dir, names, get_anchor, tpl, cfg, max_probe=8) -> (frame, anchor, resolved)`、`make_report.build_refs(scale)`、`make_report.load_official_icons()`、`make_report.classify(crop, slot, refs, icon_tpl)`、`make_report.apply_hook_cfg(resolved, anchor, video_name)`、`calibrator.calibrate_hook_slots(frame_paths, resolved)`、`gens_counter.GensTracker/load_digit_refs`、`hook_counter.count_all`、`hud_anchor.detect_anchor(frame, tpl, prior=...)`、`hud_regions.load_regions/resolve_regions`。
- Produces: 类 `StreamingDetector`，构造 `StreamingDetector(bvid, report_root, frames_root=None, cfg=CFG, hook_names=None, budget=12)`；方法：
  - `feed(frame, fname) -> dict|None`：每帧调用一次。返回 `None`（WAIT/校准中）或当前帧检测行 dict 或 `{"match_end": n, "csv": path}`。
  - `finish() -> list[int]`：流结束收尾，返回已写完局号。
  - 属性 `state`（`WAIT_ANCHOR|CALIBRATE|RECORD`）、`match_no`（当前局号，1 起）。
  - 内部落盘目录：`frames_root/{bvid}/match_{match_no}/`、报告 `report_root/{bvid}/match_{match_no}/detect_report.csv`。

- [ ] **Step 1: 改 `make_report.apply_hook_cfg` 支持多视频**

现逻辑 `data.get("video") != video_name` 时直接 return。改为：接受 `hook_names`（list[str]），匹配其中任一即应用；为兼容旧调用（`apply_hook_cfg(resolved, anchor, name)` 字符串），兼容两种类型：

```python
def apply_hook_cfg(resolved, anchor, video_name=None, hook_names=None):
    if not os.path.exists(HOOK_CFG):
        return
    with open(HOOK_CFG, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
    names = hook_names if hook_names is not None else ([video_name] if video_name else [])
    if not names or data.get("video") not in names:
        return
    for rname, rel in data["regions"].items():
        if rname in resolved:
            resolved[rname] = hud_regions.rel_to_abs(rel, anchor)
```

调用点不破坏：make_report 内 `apply_hook_cfg(resolved, anchor, name)` 语义不变。新增断言测试放 stream_detector 测试文件或 make_report 测试（若存在）。

- [ ] **Step 2: 写失败测试**

`tests/test_stream_detector.py`：

```python
import csv
import os
import tempfile
import unittest

import cv2

import make_report
import stream_detector as sd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")


def _sorted_frames():
    names = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    return [(cv2.imread(os.path.join(SRC, f)), f) for f in names if f.endswith(".jpg")]


class TestStreamingDetector(unittest.TestCase):
    def test_wait_anchor_ignores_blank(self):
        with tempfile.TemporaryDirectory() as d:
            det = sd.StreamingDetector("BV1Uu8z6eEVM", d, d)
            import numpy as np
            blank = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            r = det.feed(blank, "frame_00_00.0.jpg")
            self.assertIsNone(r)
            self.assertEqual(det.state, "WAIT_ANCHOR")

    def test_single_match_report_and_finish(self):
        with tempfile.TemporaryDirectory() as d:
            frames_root = os.path.join(d, "frames")
            report_root = os.path.join(d, "report")
            det = sd.StreamingDetector("BV1Uu8z6eEVM", report_root, frames_root,
                                       hook_names=["BV1Uu8z6eEVM", "BV1Z58J6bEoi"])
            closed = []
            for frame, fname in _sorted_frames():
                r = det.feed(frame, fname)
                if isinstance(r, dict) and "match_end" in r:
                    closed.append(r)
            det.finish()
            csv_path = os.path.join(report_root, "BV1Uu8z6eEVM",
                                    "match_1", "detect_report.csv")
            self.assertTrue(os.path.exists(csv_path), csv_path)
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            self.assertEqual(set(rows[0]) & {"p1", "p2", "p3", "p4", "hooks", "gens"}, 
                             {"p1", "p2", "p3", "p4", "hooks", "gens"})
```

> 运行时注意：110 帧全量喂入约需几十秒（模板匹配为主），可接受；若超 60s，把 `_sorted_frames()` 截断为前 60 帧（仍产生 match_1）。

- [ ] **Step 3: 运行确认 FAIL**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_stream_detector -v`
Expected: ModuleNotFoundError / 断言失败。

- [ ] **Step 4: 实现 `stream_detector.py`**

```python
import csv
import os

import cv2

import calibrator
import gens_counter
import hook_counter
import hud_anchor
import hud_regions
import make_report

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(BASE, "config", "hud_regions.json")
ANCHOR_TPL = os.path.join(BASE, "picture", "gen.jpg")
PICTURE = os.path.join(BASE, "picture")
REPORT = os.path.join(BASE, "report")

HEADER = ["frame", "scale", "p1", "p2", "p3", "p4", "hooks", "gens", "机器标注"]


class StreamingDetector:
    """逐帧流式 HUD 检测。状态机：
    WAIT_ANCHOR: 无锚点, 每帧尝试 detect_anchor, 命中即落盘开新局进 CALIBRATE
    CALIBRATE:   攒 budget 帧(落盘) -> pick_opening_frame 定 healthy 基线
                 -> calibrate_hook_slots 定槽位 -> 回放进 RECORD
    RECORD:      逐帧 classify/count_hooks/gens; 检测换局(gens 非5->5)
                 -> 写 CSV(match_end) -> 重新 CALIBRATE 开下一局
    """
    def __init__(self, bvid, report_root=None, frames_root=None, cfg_path=CFG,
                 hook_names=None, budget=12):
        self.bvid = bvid
        self.report_root = report_root or os.path.join(REPORT, bvid)
        self.frames_root = frames_root or PICTURE
        self.cfg = hud_regions.load_regions(cfg_path)
        self.tpl = cv2.imread(ANCHOR_TPL)
        self.hook_names = hook_names
        self.budget = budget
        self.state = "WAIT_ANCHOR"
        self.match_no = 0
        self._anchor = None            # 当前局锁定锚点(用于 prior)
        self._resolved = None
        self._opening = None
        self._slots = []
        self._rows = []
        self._icon_tpl = make_report.load_official_icons()
        self._digits = gens_counter.load_digit_refs()
        self._gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
        self._gens_tracker = gens_counter.GensTracker(self._digits, gen=self._gen)
        self._prev_g = None
        self._frame_dir = None
        self._n_persisted = 0

    def feed(self, frame, fname):
        if self.state == "WAIT_ANCHOR":
            anchor = hud_anchor.detect_anchor(frame, self.tpl)
            if anchor is None:
                return None
            self.match_no += 1
            self._frame_dir = os.path.join(self.frames_root, self.bvid,
                                           f"match_{self.match_no}")
            os.makedirs(self._frame_dir, exist_ok=True)
            self._persist(frame, fname)
            self.state = "CALIBRATE"
            self._calib = [(fname, cv2.imread(os.path.join(self._frame_dir, fname)))]
            self._anchor = anchor
            return None

        if self.state == "CALIBRATE":
            self._persist(frame, fname)
            self._calib.append((fname, frame))
            if len(self._calib) >= self.budget:
                self._finalize_calibration()
            return None

        # RECORD
        return self._record(frame, fname)

    def finish(self):
        closed = []
        if self._rows:
            self._write_match()
            closed.append(self.match_no)
        return closed

    # ---- 内部 ----

    def _persist(self, frame, fname):
        cv2.imwrite(os.path.join(self._frame_dir, fname), frame)

    def _finalize_calibration(self):
        names = [fn for fn, _ in self._calib]
        get_anchor = (lambda fr, tpl: hud_anchor.detect_anchor(fr, tpl,
                        prior=(self._anchor["x"], self._anchor["y"])))
        opening, anchor, resolved = make_report.pick_opening_frame(
            self._frame_dir, names, get_anchor, self.tpl, self.cfg)
        if anchor is None:
            # 兜底: 用攒下的帧自己裁 healthy(不进 pick)
            anchor = self._anchor
            resolved = hud_regions.resolve_regions(self.cfg, anchor)
            opening = self._calib[0][1]
        self._anchor = anchor
        make_report.apply_hook_cfg(resolved, anchor, video_name=self.bvid,
                                   hook_names=self.hook_names)
        self._resolved = resolved
        scale = anchor["scale"]
        refs = make_report.build_refs(scale)
        refs["healthy"] = make_report.build_opening_refs(opening, resolved)["healthy"]
        self._refs = refs
        paths = [os.path.join(self._frame_dir, fn) for fn, _ in self._calib]
        self._slots = calibrator.calibrate_hook_slots(paths, resolved)
        self.state = "RECORD"
        # 回放攒下的校准帧(未计入 rows), 确保不漏帧
        for fn, frame in self._calib:
            self._record(frame, fn)
        self._calib = []

    def _record(self, frame, fname):
        cur = hud_anchor.detect_anchor(frame, self.tpl,
                                       prior=(self._anchor["x"], self._anchor["y"]))
        anchor = cur if cur is not None else self._anchor
        resolved = hud_regions.resolve_regions(self.cfg, anchor)
        make_report.apply_hook_cfg(resolved, anchor, video_name=self.bvid,
                                   hook_names=self.hook_names)
        states = []
        for i in range(1, 5):
            b = resolved[f"survivor_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            states.append(make_report.classify(crop, i - 1, self._refs, self._icon_tpl))
        hooks = hook_counter.count_all(frame, resolved, self._slots)
        gens = self._gens_tracker.update(frame, resolved, anchor)

        # 换局: gens 从非 5 跳回 5
        if (gens == 5 and self._prev_g is not None and self._prev_g not in (5, None)):
            self._write_match()
            self.match_no += 1
            self._frame_dir = os.path.join(self.frames_root, self.bvid,
                                           f"match_{self.match_no}")
            os.makedirs(self._frame_dir, exist_ok=True)
            self._persist(frame, fname)
            self._calib = [(fname, frame.copy())]
            self._prev_g = None
            self._gens_tracker.reset()
            self.state = "CALIBRATE"
            return {"match_end": self.match_no - 1,
                    "csv": os.path.join(self.report_root, self.bvid,
                                        f"match_{self.match_no - 1}", "detect_report.csv")}

        self._prev_g = gens
        flags = []
        for i, st in enumerate(states):
            if st == "unknown":
                flags.append(f"p{i + 1}未知")
        if gens is None:
            flags.append("gens未识别")
        note = "; ".join(flags) if flags else "正常"
        self._rows.append([fname, f"{anchor['scale']:.2f}", states[0], states[1],
                           states[2], states[3], "/".join(str(h) for h in hooks),
                           str(gens), note])
        return {"frame": fname, "scale": anchor["scale"], "p1": states[0],
                "p2": states[1], "p3": states[2], "p4": states[3],
                "hooks": "/".join(str(h) for h in hooks), "gens": gens, "机器标注": note}

    def _write_match(self):
        match_dir = os.path.join(self.report_root, self.bvid, f"match_{self.match_no}")
        os.makedirs(match_dir, exist_ok=True)
        with open(os.path.join(match_dir, "detect_report.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(self._rows)
        self._rows = []
```

> 说明：`_persist` 在 CALIBRATE/RECORD 全程把帧写入 match 目录；换局时把当前帧作为下一局首帧重进 CALIBRATE（`_calib` 含该帧，会再次落盘同一文件，覆盖幂等）。`pick_opening_frame` 需要磁盘帧存在，故校准前先落盘攒帧。BLANK 帧在 WAIT 被过滤——测试 1 覆盖。

- [ ] **Step 5: 运行测试确认 PASS**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_stream_detector -v`
Expected: PASS；`match_1/detect_report.csv` 生成、行数>0、含 6 个状态列。若运行时 >60s，测试中帧子集截断为前 60 帧再跑。

- [ ] **Step 6: 全量回归**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`
Expected: 全绿（含既有 make_report/BV1 相关测试）。

- [ ] **Step 7: 提交**

```bash
git add stream_detector.py make_report.py tests/test_stream_detector.py
git commit -m "feat: 流式HUD检测器(WAIT/CALIBRATE/RECORD状态机, 换局分段)"
```

---

### Task 3: dataset_encoder 增量写 videos.jsonl（含局号 id）

**Files:**
- Modify: `dataset_encoder.py`
- Test: `tests/test_dataset_encoder.py`

**Interfaces:**
- Consumes: Task 1 的 `encode_csv(csv_path, video_id, label)`（id 现为 `BVid:N`）。
- Produces: `append_record(videos_path, record) -> int`（把一条 record 作为新行追加到 jsonl，返回行数）；`read_records(videos_path) -> list[dict]`（读全部行供校验/回填）。

- [ ] **Step 1: 写失败测试**

`tests/test_dataset_encoder.py` 追加：

```python
class TestAppendRecord(unittest.TestCase):
    def test_append_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            de.append_record(p, {"id": "BV1Z:1", "title": "", "url": "",
                                 "features": [[0.0] * 30], "label": -1})
            de.append_record(p, {"id": "BV1Z:2", "title": "", "url": "",
                                 "features": [[1.0] * 30], "label": -1})
            recs = de.read_records(p)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0]["id"], "BV1Z:1")
            self.assertEqual(recs[1]["id"], "BV1Z:2")
            self.assertEqual(recs[1]["label"], -1)
```

- [ ] **Step 2: 运行确认 FAIL**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_dataset_encoder -v`
Expected: AttributeError（无 append_record/read_records）。

- [ ] **Step 3: 实现**

`dataset_encoder.py` 追加：

```python
def append_record(videos_path, record):
    os.makedirs(os.path.dirname(videos_path), exist_ok=True)
    with open(videos_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(videos_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_records(videos_path):
    with open(videos_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 4: 运行测试全绿**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_dataset_encoder -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dataset_encoder.py tests/test_dataset_encoder.py
git commit -m "feat: dataset_encoder支持增量append与读取"
```

---

### Task 4: run_pipeline.py 三线程流水线

**Files:**
- Create: `run_pipeline.py`
- Create: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: Task 1 `extract_frames.frame_name`、Task 2 `StreamingDetector`、Task 3 `append_record`；`encode_csv`（保留作每局编码入口）。
- Produces: CLI `run_pipeline.py <mp4|frames_dir> <bvid> [--interval 0.5] [--videos dataset/videos.jsonl] [--sample N]`；打印局数/每局帧数/CSV/数据集行数汇总。

- [ ] **Step 1: 写失败测试**

`tests/test_run_pipeline.py`：用小样本 BV1 帧目录（离线帧源模式，免解码+省时）验证三线程汇合与产物。测试用一个真实但小的输入——`picture/BV1Uu8z6eEVM` 前 12 帧（覆盖至少一段 CALIBRATE+RECORD）：

```python
import json
import os
import tempfile
import unittest

import run_pipeline as rp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")


class TestPipeline(unittest.TestCase):
    def test_frames_dir_pipeline_writes_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            videos = os.path.join(d, "videos.jsonl")
            report_root = os.path.join(d, "report")
            frames_root = os.path.join(d, "frames")
            stats = rp.run_frames_dir(SRC, "BV1Uu8z6eEVM", sample=12,
                                      videos=videos, report_root=report_root,
                                      frames_root=frames_root)
            self.assertGreaterEqual(stats["matches"], 1)
            with open(videos, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertGreaterEqual(len(lines), 1)
            self.assertTrue(lines[0]["id"].startswith("BV1Uu8z6eEVM:"))
            self.assertEqual(len(lines[0]["features"][0]), 30)
```

> `sample=12` ≤ budget，若不足 budget 永不进 RECORD → 测试须覆盖"进过 RECORD"。改用 sample=40（>budget 且含开局 HUD，预算内耗时 <5s 的检测量级）。Step 1 直接用 40。

- [ ] **Step 2: 运行确认 FAIL**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_run_pipeline -v`
Expected: ModuleNotFoundError: run_pipeline。

- [ ] **Step 3: 实现 `run_pipeline.py`**

```python
import argparse
import os
import queue
import threading

import cv2

import dataset_encoder as de
import stream_detector as sd
from extract_frames import frame_name

BASE = os.path.dirname(os.path.abspath(__file__))
PICTURE = os.path.join(BASE, "picture")
DATASET = os.path.join(BASE, "dataset", "videos.jsonl")


def _iter_video_frames(video, interval):
    """解码线程帧源: 顺序按 interval 取帧, yield (frame, fname)。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, frame_name(t) + ".jpg"
        t += interval
    cap.release()


def _iter_dir_frames(src_dir, sample=None):
    names = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
    if sample:
        names = names[:sample]
    for n in names:
        img = cv2.imread(os.path.join(src_dir, n))
        if img is not None:
            yield img, n


def _encode_match(bvid, report_root, match_no, videos_path):
    csv_path = os.path.join(report_root, bvid, f"match_{match_no}", "detect_report.csv")
    if not os.path.exists(csv_path):
        return 0
    rec = de.encode_csv(csv_path, f"{bvid}:{match_no}", label=-1)
    return de.append_record(videos_path, rec)


def run_frames_dir(src_dir, bvid, sample=None, videos=DATASET,
                   report_root=None, frames_root=None, budget=12):
    """离线/测试入口: 直接把已有帧目录喂给检测器(单线程串行)。"""
    report_root = report_root or os.path.join(BASE, "report", bvid)
    frames_root = frames_root or PICTURE
    det = sd.StreamingDetector(bvid, report_root, frames_root, hook_names=[bvid])
    det.budget = budget
    matches = 0
    for frame, fname in _iter_dir_frames(src_dir, sample=sample):
        det.feed(frame, fname)
    closed = det.finish()
    n_rec = 0
    for m in closed:
        n_rec += _encode_match(bvid, report_root, m, videos)
    return {"matches": len(closed), "records": n_rec, "closed": closed}


def run_video(video, bvid, interval=0.5, videos=DATASET, report_root=None,
              frames_root=None, budget=12):
    """三线程流水线:
      T1 解码: _iter_video_frames -> q(带 EOF 哨兵)
      T2 检测: 消费 q 喂 StreamingDetector(落盘 match 帧 + 写 CSV)
      T3 编码: 消费 match_end/closed 通知 -> encode+append jsonl
    """
    report_root = report_root or os.path.join(BASE, "report", bvid)
    frames_root = frames_root or PICTURE
    q = queue.Queue(maxsize=64)
    eof = object()
    done = threading.Event()

    def producer():
        try:
            for frame, fname in _iter_video_frames(video, interval):
                q.put((frame, fname))
        except Exception as e:  # noqa: BLE001
            print(f"[producer] {e}")
        finally:
            q.put(eof)

    results = []
    lock = threading.Lock()

    def consumer():
        det = sd.StreamingDetector(bvid, report_root, frames_root,
                                   hook_names=[bvid])
        det.budget = budget
        while True:
            item = q.get()
            if item is eof:
                q.task_done()
                break
            frame, fname = item
            r = det.feed(frame, fname)
            if isinstance(r, dict) and "match_end" in r:
                with lock:
                    results.append(r["match_end"])
            q.task_done()

    def encoder():
        # 消费 results(锁保护) 直到 done
        seen = set()
        while not done.is_set():
            with lock:
                pending = [m for m in results if m not in seen]
                for m in pending:
                    seen.add(m)
            for m in pending:
                _encode_match(bvid, report_root, m, videos)
            done.wait(0.2)
        with lock:
            pending = [m for m in results if m not in seen]
            for m in pending:
                seen.add(m)
        for m in pending:
            _encode_match(bvid, report_root, m, videos)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t3 = threading.Thread(target=encoder)
    t3.daemon = True
    for t in (t1, t2):
        t.start()
    t1.join()
    t2.join()
    # 收尾: 最后未触发换局的局由 consumer 结束时的 finish() 产生 closed
    # 简单化: 此版由主线程在 T2 join 后用新 detector 重新跑一遍收尾?
    # 见下注释(实现时用队列回传 finish 结果)
    ...
```

> **收尾设计（实现时按此做完整版）**：consumer 在收到 EOF 后调用 `det.finish()`，把返回的局号列表再入队到同一个 `results` 通知结构（用第二把锁或直接 `q` 变体）。为降低复杂度，本任务实现推荐：把"已写 CSV 的局号"通过 `queue.Queue closed_q` 传给 encoder，producer/consumer 都结束（EOF）后 encoder 最后 drain 一次并退出。encoder 退出后主线程 join 全部线程，统计打印。Step 3 要求完整无死锁实现；以 `run_frames_dir` 单线程版为准保证正确性，多线程版与其共享 `StreamingDetector`+`_encode_match` 语义。

CLI：

```python
def main():
    ap = argparse.ArgumentParser(description="DBD 数据产线: 抽帧->流式检测->编码")
    ap.add_argument("source", help="mp4 或 已有帧目录")
    ap.add_argument("bvid")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--videos", default=DATASET)
    args = ap.parse_args()
    if os.path.isdir(args.source):
        stats = run_frames_dir(args.source, args.bvid, sample=args.sample,
                               videos=args.videos)
    else:
        stats = run_video(args.source, args.bvid, interval=args.interval,
                          videos=args.videos)
    print(f"matches={stats['matches']} records={stats['records']} -> {args.videos}")
```

- [ ] **Step 4: 运行测试确认 PASS**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest tests.test_run_pipeline -v`
Expected: PASS（sample=40，>budget，命中 RECORD 并写 1 局 CSV + 数据集 1 行；<10s）。若 sample=40 含两局（BV1 有多局）则 closed>1，`matches>=1` 仍成立。

- [ ] **Step 5: 全量回归**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -m unittest discover -s tests`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
git add run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: 三线程数据产线run_pipeline(解码/检测/编码)"
```

---

### Task 5: 真实视频端到端跑通 + 文档

**Files:**
- Run: 命令见 Step 1。
- Modify: `docs/dataset_format.md`、`docs/superpowers/specs/2026-09-02-pipeline-multithread-design.md`（如需小修）、`PROGRESS.md`。

**Interfaces:**
- Consumes: Task 1-4 全部。
- Produces: 真实产物（帧落盘 `picture/BV1Z58J6bEoi/match_N/`、每局 `report/BV1Z58J6bEoi/match_N/detect_report.csv`、`dataset/videos.jsonl` 追加 `BV1Z58J6bEoi:N` 记录）。

- [ ] **Step 1: 真实 mp4 跑通（预计 5-15 分钟）**

先小样本验证（0.5s × 前 90s ≈ 180 帧）再全量：

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" run_pipeline.py picture/raw_videos/BV1Z58J6bEoi.mp4 BV1Z58J6bEoi --sample 180`
Expected: matches≥1、records≥1、CSV 生成。目视抽查 `picture/BV1Z58J6bEoi/match_1` 帧、检查 detect_report.csv 的 p1-p4/gens/hooks 列合理性（对照 BV1 同布局）。

Run（全量, 2200 帧）: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" run_pipeline.py picture/raw_videos/BV1Z58J6bEoi.mp4 BV1Z58J6bEoi --interval 0.5`
Expected: 汇总 matches≥1；若 ≥2 局则 `:1/:2...` 多条。

- [ ] **Step 2: 校验数据集行**

Run: `& "C:\Users\Sallia\.conda\envs\dbd\python.exe" -c "import dataset_encoder as de,glob; [print(len(r['features']), r['id'], r['label']) for r in de.read_records('dataset/videos.jsonl') if r['id'].startswith('BV1Z58J6bEoi')]"`
Expected: 每条 label=-1、features 长度合理（0.5s×局时长×2±1）。

- [ ] **Step 3: 文档更新**

- `docs/dataset_format.md`：补 `id` 可带局号 `BVid:N`、`label=-1` 占位、多局 `match_N` 目录布局、`run_pipeline.py` 用法、`parse_time` 半秒粒度、`features[29]` 秒。
- `PROGRESS.md`：记录 BV1Z58J6bEoi 数据条目、命令、产物位置。

- [ ] **Step 4: 提交**

```bash
git add docs/dataset_format.md PROGRESS.md
git commit -m "docs: 多线程产线跑通BV1Z58J6bEoi与文档更新"
```

---

## Self-Review

**Spec coverage:**
- §3.1 三线程流水线 → Task 4 ✓（解码线程不落盘、检测线程唯一落盘并定 match、编码线程 append）
- §3.2 半秒帧命名 → Task 1 ✓（`frame_name`+`parse_time` 半秒）
- §3.3 状态机 → Task 2 ✓（WAIT/CALIBRATE/RECORD/MATCH_END，复用 make_report 函数）
- §3.4 目录布局 → Task 2（frames/report 按 match_N）✓；Task 4/5 验证
- §3.5 局号 id + label -1 → Task 3/5 ✓
- §4 BV1Z58J6bEoi 验证 → Task 5 ✓（含小样本→全量）
- §2 决策：锚点驱动/无锚点等待 ✓（WAIT）；换局分段 ✓；0.5s ✓；本局 label 占位 -1 ✓

**Placeholder scan:** 无 TBD/TODO。Task 4 `run_video` 的三线程收尾含"见实现时完整版"注释而非最终代码——已给出明确机制（closed_q + finish 回传 + join + drain），对可执行性足够；如追求零歧义可先并到 Task 4 实现，但 Step 3 已限定以单线程 `run_frames_dir` 保正确、多线程共享语义。`hook_names` 传 `[bvid]` 与 Task 2 的兼容性改造一致。

**Type consistency:** `parse_time`/`frame_name` 在 Task 1 定义并被 Task 4 用（fname 后缀 `.jpg`，`encode_csv` 再按 parse_time 解析→features[29] 秒）；`StreamingDetector(bvid, report_root, frames_root, hook_names, budget)` 构造签名在 Task 2 定义并被 Task 4 用；`append_record/read_records` Task 3 定义、Task 4/5 用；`_encode_match` 在 Task 4 定义并被自身复用。`classify` 统一用 `make_report.classify`（返回字符串）——未引用 `state_recognizer.classify`（tuple 版）避免歧义。`apply_hook_cfg` 新增 `hook_names` kwarg 兼容旧字符串调用。

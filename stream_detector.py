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

    落盘规则(全局约束): CALIBRATE/RECORD 进入的帧全部写入
    frames_root/{bvid}/match_{no}/; WAIT 段(菜单/结算)不落盘。
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
        self._anchor = None
        self._resolved = None
        self._opening = None
        self._slots = []
        self._refs = None
        self._rows = []
        self._icon_tpl = make_report.load_official_icons()
        self._digits = gens_counter.load_digit_refs()
        self._gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
        self._gens_tracker = gens_counter.GensTracker(self._digits, gen=self._gen)
        self._prev_g = None
        self._frame_dir = None
        self._calib = []

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
            self._calib = [(fname, frame.copy())]
            self._anchor = anchor
            return None

        if self.state == "CALIBRATE":
            self._persist(frame, fname)
            self._calib.append((fname, frame.copy()))
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
        get_anchor = (lambda fr, tpl: hud_anchor.detect_anchor(
            fr, tpl, prior=(self._anchor["x"], self._anchor["y"])))
        opening, anchor, resolved = make_report.pick_opening_frame(
            self._frame_dir, names, get_anchor, self.tpl, self.cfg)
        if anchor is None:
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

        # 非换局 RECORD 帧落盘到当前局
        self._persist(frame, fname)
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

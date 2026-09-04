import json
import os

import cv2
import numpy as np

import hud_regions

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "picture", "test1")
CFG = os.path.join(BASE, "config", "hud_regions.json")
GEN_TPL = os.path.join(BASE, "picture", "gen.jpg")
DIGIT_REFS_DIR = os.path.join(BASE, "asset", "gens_digits")
DIGIT_REFS_JSON = os.path.join(DIGIT_REFS_DIR, "refs.json")

DIGIT_X = 29
# 跨视频实测: 有 gen 图标帧 NCC≥0.66 (BV1pht96fEjN 最低 0.66), "全修完/大门"帧 NCC≤0.28。
# 旧值 0.70 会让 NCC≈0.66~0.69 的真实图标误入"消失→0"分支。
GEN_ICON_THR = 0.55
DIGIT_THR = 0.55
LOW_THR = 0.45
TRACK_NCC = 0.85

DIGIT_REFS_SRC = {
    5: ["frame_0001"],
    4: ["frame_0002", "frame_0003", "frame_0004", "frame_0005"],
    3: ["frame_0006"],
    2: ["frame_0007", "frame_0008", "frame_0009"],
}

def _ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)

def _match_gen_icon(crop, gen, anchor):
    s = anchor["scale"]
    th, tw = gen.shape[:2]
    w, h = int(tw * s), int(th * s)
    t = cv2.resize(gen, (w, h), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return float(maxv)

def load_digit_refs():
    """从 asset/gens_digits 加载通用高清数字模板库。返回 {digit: [img,...]}。"""
    with open(DIGIT_REFS_JSON) as f:
        meta = json.load(f)
    refs = {}
    for digit, files in meta.items():
        imgs = []
        for fn in files:
            path = os.path.join(DIGIT_REFS_DIR, fn)
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)
        if imgs:
            refs[int(digit)] = imgs
    return refs

def build_digit_refs():
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    return _build_refs(resolved, SRC, DIGIT_REFS_SRC, scale=1.0)

def build_digit_refs_from_video(frame_dir, resolved, digit_frames, scale):
    """从指定视频按显式帧映射构建数字参考（数字宽度=DIGIT_X*scale）。"""
    return _build_refs(resolved, frame_dir, digit_frames, scale)

def _build_refs(resolved, frame_dir, digit_frames, scale):
    b = resolved["gens_row"]
    digit_w = int(DIGIT_X * scale)
    refs = {}
    for digit, frames in digit_frames.items():
        imgs = []
        for f in frames:
            path = os.path.join(frame_dir, f + ".jpg")
            if not os.path.exists(path):
                path = os.path.join(frame_dir, f + ".0.jpg")
            frame = cv2.imread(path)
            if frame is None:
                continue
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            imgs.append(crop[:, 0:digit_w])
        if imgs:
            refs[digit] = imgs
    return refs

def count_gens(frame, resolved, refs, gen=None, anchor=None, gen_icon_thr=GEN_ICON_THR, digit_thr=DIGIT_THR):
    b = resolved["gens_row"]
    crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if int((g > 100).sum()) < 20:
        return None

    if gen is None:
        gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
    if anchor is None:
        anchor = {"scale": 1.0}

    if _match_gen_icon(crop, gen, anchor) >= gen_icon_thr:
        digit_w = int(DIGIT_X * anchor["scale"])
        digit_crop = crop[:, 0:digit_w]
        dh, dw = digit_crop.shape[:2]

        def fit(img):
            if img.shape[:2] != (dh, dw):
                return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            return img

        best, best_score = None, -1.0
        for digit, imgs in refs.items():
            for ref in imgs:
                s = _ncc(digit_crop, fit(ref))
                if s > best_score:
                    best_score, best = s, digit
        if best is not None and best_score >= digit_thr:
            return best
        return None
    return 0


class GensTracker:
    """状态化 gens 数字识别器。

    利用 gens 数字的时序特性提升鲁棒性：
    1. 帧间数字框 NCC 高 -> 沿用前一帧结果（免模板匹配）
    2. 模板重识别
    3. 递减约束：识别结果大于前一帧时沿用前一帧（gens 只减不增）

    状态（prev_digit / prev_crop）跨帧保留，换局时调用 reset()。
    """

    def __init__(self, refs, gen=None, icon_thr=GEN_ICON_THR,
                 digit_thr=DIGIT_THR, track_ncc=TRACK_NCC, low_thr=LOW_THR):
        self.refs = refs
        self.gen = gen if gen is not None else cv2.imread(GEN_TPL)
        self.icon_thr = icon_thr
        self.digit_thr = digit_thr
        self.track_ncc = track_ncc
        self.low_thr = low_thr
        self.prev_digit = None
        self.prev_crop = None

    def reset(self):
        self.prev_digit = None
        self.prev_crop = None

    def update(self, frame, resolved, anchor):
        """处理单帧，返回 gens 数字（1-5 / 0 / None）。状态跨帧保留。"""
        b = resolved["gens_row"]
        crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if int((g > 100).sum()) < 20:
            self.prev_digit = None
            self.prev_crop = None
            return None

        if anchor is None:
            anchor = {"scale": 1.0}

        if _match_gen_icon(crop, self.gen, anchor) >= self.icon_thr:
            digit_w = int(DIGIT_X * anchor["scale"])
            digit_crop = crop[:, 0:digit_w]

            # 1) 帧间沿用：与前一帧数字框高度相似则沿用结果
            if (self.prev_crop is not None and self.prev_digit is not None):
                sim = _ncc(digit_crop, self.prev_crop)
                if sim >= self.track_ncc:
                    self.prev_crop = digit_crop
                    return self.prev_digit

            # 2) 模板重识别
            dh, dw = digit_crop.shape[:2]
            best, best_score = None, -1.0
            for digit, imgs in self.refs.items():
                for ref in imgs:
                    r = ref
                    if r.shape[:2] != (dh, dw):
                        r = cv2.resize(r, (dw, dh), interpolation=cv2.INTER_AREA)
                    s = _ncc(digit_crop, r)
                    if s > best_score:
                        best_score, best = s, digit

            if best is not None and best_score >= self.digit_thr:
                # 3) 递减约束：gens 只减不增，识别结果大于前帧则沿用前帧
                if self.prev_digit is not None and best > self.prev_digit:
                    self.prev_crop = digit_crop
                    return self.prev_digit
                self.prev_digit = best
                self.prev_crop = digit_crop
                return best

            # 4) 模板分数不足：best 仍明确(>=LOW_THR)且符合递减则采纳；
            #    否则若前一帧有效则沿用（防御渲染噪声）
            if (best is not None and best_score >= self.low_thr
                    and (self.prev_digit is None or best <= self.prev_digit)):
                self.prev_digit = best
                self.prev_crop = digit_crop
                return best
            if self.prev_digit is not None and self.prev_digit in (1, 2, 3, 4, 5):
                self.prev_crop = digit_crop
                return self.prev_digit
            self.prev_digit = None
            self.prev_crop = None
            return None

        # 图标消失 -> 0（所有发电机修完）
        self.prev_digit = 0
        self.prev_crop = None
        return 0

def main():
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    refs = load_digit_refs()
    gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))

    truth = {
        "frame_0000": None, "frame_0001": 5, "frame_0002": 4, "frame_0003": 4,
        "frame_0004": 4, "frame_0005": 4, "frame_0006": 3, "frame_0007": 2,
        "frame_0008": 2, "frame_0009": 2, "frame_0010": 0, "frame_0011": 0,
    }
    print(f"{'frame':12} {'gens':6} {'truth':6} {'ok'}")
    all_ok = True
    tracker = GensTracker(refs, gen=gen)
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".jpg"):
            continue
        stem = fname[:-4]
        frame = cv2.imread(os.path.join(SRC, fname))
        got = tracker.update(frame, resolved, cfg["anchor"])
        exp = truth.get(stem)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"{stem:12} {str(got):6} {str(exp):6} {'OK' if ok else 'FAIL'}")
    print("ALL OK" if all_ok else "SOME FAILED")

if __name__ == "__main__":
    main()

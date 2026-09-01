import os

import cv2
import numpy as np

CROPS_DIR = r"D:\files\code\dbd_pred\picture\crops"

ICON_STATES = ["hooked", "dying", "dead", "escaped"]
FACE_STATES = ["healthy", "injured"]

EXAMPLE_REFS = {
    "hooked":  [("frame_0003", "survivor_p3")],
    "dying":   [("frame_0003", "survivor_p4")],
    "dead":    [("frame_0009", "survivor_p4")],
    "escaped": [("frame_0011", "survivor_p3")],
    "healthy": [("frame_0001", "survivor_p1"),
                ("frame_0001", "survivor_p2"),
                ("frame_0001", "survivor_p3"),
                ("frame_0001", "survivor_p4")],
    "injured": [("frame_0002", "survivor_p4")],
}

def load_crop(frame_stem, region):
    return cv2.imread(os.path.join(CROPS_DIR, frame_stem, f"{region}.jpg"))

def build_references():
    refs = {}
    for state, pairs in EXAMPLE_REFS.items():
        imgs = [load_crop(f, r) for f, r in pairs]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            refs[state] = imgs
    return refs

def ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)

ICON_SCALES = 10


def red_diag_lines(crop, healthy_ref, diff_thr=40, minlen=5, angle_tol=25, diag_thr=6):
    """检测受伤红斜线：与健康参考做 R 通道差分，统计 -45 度斜线数量。"""
    c = crop.astype(np.float32)
    b = healthy_ref.astype(np.float32)
    h, w = crop.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    diff = c[:, :, 2] - b[:, :, 2]
    m = (diff > diff_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=6,
                            minLineLength=minlen, maxLineGap=2)
    diag = 0
    total = 0
    if lines is not None:
        total = len(lines)
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a + 45) < angle_tol or abs(a - 135) < angle_tol:
                diag += 1
    return int(m.sum() / 255), diag, total


def red_diag_long(crop, healthy_ref, diff_thr=40, minlen=10, angle_tol=20):
    """统计长度>=minlen 的完整 -45 度斜线（跨越整幅头像的受伤条纹）。"""
    c = crop.astype(np.float32)
    b = healthy_ref.astype(np.float32)
    h, w = crop.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    diff = c[:, :, 2] - b[:, :, 2]
    m = (diff > diff_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=6,
                            minLineLength=minlen, maxLineGap=3)
    long_d = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a + 45) < angle_tol or abs(a - 135) < angle_tol:
                if int(np.hypot(x2 - x1, y2 - y1)) >= minlen:
                    long_d += 1
    return long_d


def _hist_match(img, ref):
    """逐通道直方图匹配：消除整体亮度/对比度差异（雾、光照变化）。"""
    out = np.zeros_like(img)
    for ch in range(3):
        src = img[:, :, ch].ravel()
        dst = ref[:, :, ch].ravel()
        s_hist, _ = np.histogram(src, 256, [0, 256])
        d_hist, _ = np.histogram(dst, 256, [0, 256])
        s_cdf = s_hist.cumsum() / s_hist.sum()
        d_cdf = d_hist.cumsum() / d_hist.sum()
        map_ = np.interp(s_cdf, d_cdf, np.arange(256))
        out[:, :, ch] = map_[src].reshape(img[:, :, ch].shape)
    return out.astype(np.uint8)


def red_diag_clusters(crop, healthy_ref, diff_thr=40, minlen=10, angle_tol=15,
                      rho_tol=8, center=(0.7, 0.8), yratio=0.5):
    """检测受伤红斜线：均值对齐后中心区域 -45 度斜线，要求延伸到头像下 50%。
    受伤条纹是横跨头像下部的斜线；左上角装饰差异被 yratio 排除。
    只用 R 通道整体均值对齐（而非直方图匹配）以区分整体变亮（HUD/光照，
    R 均值均匀上升，对齐后无斜线）与局部受伤红斜线（移除整体偏移后仍显著）。
    返回簇数。"""
    c = crop.astype(np.float32)
    b = healthy_ref.astype(np.float32)
    h, w = c.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    c = c.copy()
    c[:, :, 2] -= (c[:, :, 2] - b[:, :, 2]).mean()
    diff = c[:, :, 2] - b[:, :, 2]
    m = (diff > diff_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    fx, fy = center
    x0 = int(w * (1 - fx) / 2)
    x1 = int(w * (1 + fx) / 2)
    y0 = int(h * (1 - fy) / 2)
    y1 = int(h * (1 + fy) / 2)
    sub = np.zeros_like(m)
    sub[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    lines = cv2.HoughLinesP(sub, 1, np.pi / 180, threshold=6,
                            minLineLength=minlen, maxLineGap=3)
    if lines is None:
        return 0
    bs = []
    theta = np.radians(135)
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if not (abs(a + 45) < angle_tol or abs(a - 135) < angle_tol):
            continue
        if int(np.hypot(x2 - x1, y2 - y1)) < minlen:
            continue
        if max(y1, y2) < h * yratio:
            continue
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        bs.append(rho)
    if not bs:
        return 0
    bs = np.sort(bs)
    clusters = [[bs[0]]]
    for r in bs[1:]:
        if r - clusters[-1][-1] <= rho_tol:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    return len(clusters)


def multi_scale_match(crop, icon):
    g_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_crop -= g_crop.mean()
    ih, iw = icon.shape[:2]
    best = -1.0
    for scale in np.linspace(0.3, 1.2, ICON_SCALES):
        w = int(iw * scale)
        h = int(ih * scale)
        if w < 6 or h < 6 or w > g_crop.shape[1] or h > g_crop.shape[0]:
            continue
        t = cv2.resize(icon, (w, h), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        t -= t.mean()
        res = cv2.matchTemplate(g_crop, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        if mx > best:
            best = mx
    return best


def redness(crop, frac=0.5):
    h, w = crop.shape[:2]
    my, mx = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    crop = crop[my:h - my, mx:w - mx]
    b = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "rg": float((r - g).mean()),
        "sat": float(hsv[:, :, 1].mean()),
    }

def classify(crop, slot, refs, icon_thr=0.55, face_thr=0.35, inj_rg_delta=12.0, inj_sat_delta=15.0,
             icon_tpl=None):
    best_icon, best_score = None, -1.0
    for state in ICON_STATES:
        for ref in refs.get(state, []):
            s = ncc(crop, ref)
            if s > best_score:
                best_score, best_icon = s, state
    if best_score >= icon_thr:
        return best_icon, best_score
    if icon_tpl:
        for state, tpl in icon_tpl.items():
            if not tpl:
                continue
            s = multi_scale_match(crop, tpl)
            if s > best_score:
                best_score, best_icon = s, state
        if best_score >= icon_thr + 0.15:
            return best_icon, best_score

    healthy_refs = refs.get("healthy", [])
    if 0 <= slot < len(healthy_refs):
        s = ncc(crop, healthy_refs[slot])
        if s < face_thr:
            return "unknown", s
        cur = redness(crop)
        base = redness(healthy_refs[slot])
        if cur["rg"] > base["rg"] + inj_rg_delta and cur["sat"] > base["sat"] + inj_sat_delta:
            return "injured", s
        return "healthy", s
    return "unknown", best_score

def main():
    refs = build_references()
    for state, imgs in refs.items():
        print(f"{state}: {[i.shape for i in imgs]}")
    print()
    print(f"{'frame':14} {'p1':14} {'p2':14} {'p3':14} {'p4':14}")
    for f in sorted(os.listdir(CROPS_DIR)):
        row = []
        for i in range(1, 5):
            crop = load_crop(f, f"survivor_p{i}")
            if crop is None:
                row.append("----")
                continue
            state, score = classify(crop, i - 1, refs)
            row.append(f"{state}({score:.2f})")
        print(f"{f:14} {row[0]:14} {row[1]:14} {row[2]:14} {row[3]:14}")

if __name__ == "__main__":
    main()

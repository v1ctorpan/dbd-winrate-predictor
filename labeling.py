"""结局自动标注：由一局末尾若干帧的 4 人状态推断逃生人数(0-4)。

依据用户确认：用**最后一帧 HUD** 的 4 个头像图标判断逃生人数；escaped 为终态，
一旦出现即保持。为抗末帧 HUD 淡出/瞬时高亮，统计窗口内"是否出现过 escaped"。
"""


def infer_label(states_per_frame, window=40):
    """states_per_frame: 按时间升序的每帧 4 人状态列表，如 [["escaped","dead",...], ...]。

    返回最后 window 帧中出现过 escaped 的人数(0-4)。
    若窗口内没有任何非 unknown 的有效 HUD → 返回 None(无法判定)。
    """
    tail = states_per_frame[-window:] if window else states_per_frame
    valid = [r for r in tail if any(s != "unknown" for s in r)]
    if not valid:
        return None
    return sum(any(r[i] == "escaped" for r in valid) for i in range(4))


def infer_label_from_csv(csv_path, window=40):
    """从 detect_report.csv 读取 p1~p4 列推断结局标签；无法判定返回 None。"""
    import csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append([r.get(f"p{i}", "unknown") for i in range(1, 5)])
    if not rows:
        return None
    return infer_label(rows, window=window)

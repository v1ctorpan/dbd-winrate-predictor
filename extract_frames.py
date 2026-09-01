import os
import sys

import cv2
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))


def extract_frames(video, out_dir, interval=2, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    total_targets = int(duration // interval) + 1
    if max_frames:
        total_targets = min(total_targets, max_frames)

    t = 0.0
    extracted = 0
    pbar = tqdm(total=total_targets, desc="extracting frames", unit="frame")
    while t < duration:
        if max_frames and extracted >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        m, s = divmod(int(t), 60)
        name = f"frame_{m:02d}_{s:02d}.0"
        cv2.imwrite(os.path.join(out_dir, name + ".jpg"), frame)
        extracted += 1
        pbar.update(1)
        t += interval
    pbar.close()
    cap.release()
    return extracted


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "picture", "raw_videos", "BV1Uu8z6eEVM.mp4")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
    n = extract_frames(video, out, interval=interval, max_frames=max_frames)
    print(f"extracted {n} frames -> {out}")

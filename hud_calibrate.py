import os
import sys

import cv2

import hud_anchor
import hud_regions

BASE_REGIONS = [
    "survivor_p1", "survivor_p2", "survivor_p3", "survivor_p4",
    "hook_p1", "hook_p2", "hook_p3", "hook_p4",
    "gens_row",
]

EXTRA_REGIONS = [
    "endgame_timer",
    "gate_ui",
    "hatch_ui",
]

COLOR_POOL = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 200, 0), (255, 200, 0), (255, 200, 0), (255, 200, 0),
    (0, 200, 255), (255, 255, 0),
    (255, 0, 255), (255, 255, 255), (0, 255, 255),
]

OUT_JSON = r"D:\files\code\dbd_pred\config\hud_regions.json"
TEMPLATE_PATH = r"D:\files\code\dbd_pred\picture\gen.jpg"
DEFAULT_FRAME = r"D:\files\code\dbd_pred\picture\test1\frame_0001.jpg"
TRUST_SCORE = 0.80

class Calibrator:
    def __init__(self, frame_path, region_names, template_path=TEMPLATE_PATH):
        self.frame = cv2.imread(frame_path)
        if self.frame is None:
            raise SystemExit(f"cannot load {frame_path}")
        self.template = cv2.imread(template_path)
        self.name = os.path.basename(frame_path)
        self.region_names = region_names

        cached = None
        if os.path.exists(OUT_JSON):
            cached = hud_regions.load_regions(OUT_JSON)
        self.anchor = hud_anchor.detect_anchor(self.frame, self.template, min_score=TRUST_SCORE)
        if self.anchor is None and cached:
            self.anchor = cached["anchor"]
            print(f"no trusted anchor (score>={TRUST_SCORE}) on {self.name}, using cached {self.anchor['x']},{self.anchor['y']}")
        self.cached = cached

        self.confirmed = [None] * len(region_names)
        self.idx = 0
        self.drag_start = None
        self.drag_cur = None

        self.scale = 1.0
        h, w = self.frame.shape[:2]
        if w > 1400:
            self.scale = 1400 / w
            self.disp = cv2.resize(self.frame, (int(w * self.scale), int(h * self.scale)))
        else:
            self.disp = self.frame.copy()

    def draw(self):
        img = self.disp.copy()
        for i, name in enumerate(self.region_names):
            color = COLOR_POOL[i % len(COLOR_POOL)]
            box = self.confirmed[i]
            if box is not None:
                b = self._to_disp(box)
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
                cv2.putText(img, name, (b[0] + 2, max(12, b[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if self.anchor is not None:
            a = self.anchor
            s = self.scale
            cv2.rectangle(img, (int(a["x"] * s), int(a["y"] * s)),
                          (int((a["x"] + a["w"]) * s), int((a["y"] + a["h"]) * s)),
                          (0, 0, 255), 3)
            cv2.putText(img, f"anchor scale={a['scale']:.2f} score={a['score']:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.drag_start and self.drag_cur:
            b = self._to_disp((*self.drag_start, *self.drag_cur))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        cur = self.region_names[self.idx] if self.idx < len(self.region_names) else "SAVE"
        cv2.putText(img, f"[{self.idx}/{len(self.region_names)}] current: {cur}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        hint = "drag=LMB  confirm=N/Space  redo=R  back=B  save=S  quit=Esc"
        cv2.putText(img, hint, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("DBD HUD Calibration", img)

    def _to_disp(self, box):
        s = self.scale
        return (int(box[0] * s), int(box[1] * s), int(box[2] * s), int(box[3] * s))

    def _to_frame(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self._to_frame(x, y)
            self.drag_cur = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_cur = self._to_frame(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            x1, y1 = self._to_frame(x, y)
            if self.idx < len(self.region_names) and abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                self.confirmed[self.idx] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.drag_start = None
            self.drag_cur = None

    def confirm(self):
        if self.idx < len(self.region_names) and self.confirmed[self.idx] is None:
            print("no box drawn for current region, skipped")
        self.idx += 1
        self.drag_start = None
        self.drag_cur = None

    def save(self):
        config = self.cached if self.cached else {
            "template": TEMPLATE_PATH,
            "ref_size": [hud_regions.REF_W, hud_regions.REF_H],
            "calibrated_on": {},
            "anchor": None,
            "regions": {},
        }
        if self.anchor is not None:
            config["anchor"] = {k: self.anchor[k] for k in ("x", "y", "w", "h", "scale")}
        if not isinstance(config.get("regions"), dict):
            config["regions"] = {}
        if not isinstance(config.get("calibrated_on"), dict):
            config["calibrated_on"] = {}
        for name, box in zip(self.region_names, self.confirmed):
            if box is not None and self.anchor is not None:
                config["regions"][name] = hud_regions.abs_to_rel(box, self.anchor)
                config["calibrated_on"][name] = self.name
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        hud_regions.save_regions(OUT_JSON, config)
        print(f"saved -> {OUT_JSON}")
        print(f"regions now: {list(config['regions'].keys())}")

    def run(self):
        cv2.namedWindow("DBD HUD Calibration")
        cv2.setMouseCallback("DBD HUD Calibration", self.on_mouse)
        while True:
            self.draw()
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("n"), ord(" "), 13):
                self.confirm()
            elif key == ord("r"):
                if 0 <= self.idx < len(self.region_names):
                    self.confirmed[self.idx] = None
            elif key == ord("b"):
                self.idx = max(0, self.idx - 1)
                if self.idx < len(self.region_names):
                    self.confirmed[self.idx] = None
            elif key == ord("s"):
                self.save()
        cv2.destroyAllWindows()

def main():
    args = sys.argv[1:]
    frame_path = args[0] if args else DEFAULT_FRAME
    region_names = args[1:] if len(args) > 1 else BASE_REGIONS + EXTRA_REGIONS
    Calibrator(frame_path, region_names).run()

if __name__ == "__main__":
    main()

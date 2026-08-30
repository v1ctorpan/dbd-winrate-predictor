import json
import os
import tkinter as tk

import cv2

import hud_anchor
import hud_regions

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(BASE, "config", "hud_regions.json")
ANCHOR_TPL = os.path.join(BASE, "picture", "gen.jpg")
FRAME_DIR = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
FRAME_NAME = "frame_02_10.0.jpg"
OUT = os.path.join(BASE, "config", "hook_regions.json")

REGIONS = ["hook_p1", "hook_p2", "hook_p3", "hook_p4"]
BGR_COLORS = [(0, 0, 255), (0, 165, 255), (0, 128, 0), (255, 0, 0)]


class App:
    def __init__(self, root):
        self.root = root
        root.title("标注 hook 区域框 (BV1Uu8z6eEVM)")
        root.geometry("1100x700")

        self.cfg = hud_regions.load_regions(CFG)
        self.tpl = cv2.imread(ANCHOR_TPL)
        self.frame = cv2.imread(os.path.join(FRAME_DIR, FRAME_NAME))
        self.anchor = hud_anchor.detect_anchor(self.frame, self.tpl, prior=(121, 847))
        if self.anchor is None:
            raise SystemExit("锚点检测失败")
        self.scale = self.anchor["scale"]

        self.display = self.frame.copy()
        self.overlay = self.display.copy()
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        tip = tk.Label(
            root,
            text="鼠标拖拽移动当前玩家框 | 数字1-4切换玩家 | s保存 | r恢复初始 | q退出",
        )
        tip.pack()
        self.status = tk.Label(root, text="", anchor="w")
        self.status.pack(fill=tk.X)

        self.boxes = {}
        self.cursor = {}
        for name in REGIONS:
            b = self.cfg["regions"][name]
            self.boxes[name] = hud_regions.rel_to_abs(b, self.anchor)
            self.cursor[name] = None

        self.current = "hook_p1"
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        root.bind("<Key>", self.on_key)
        root.after(50, self.redraw)

    def refresh_overlay(self):
        self.overlay = self.display.copy()
        for i, name in enumerate(REGIONS):
            b = self.boxes[name]
            color = BGR_COLORS[i]
            sel = 3 if name == self.current else 1
            self.overlay = cv2.rectangle(self.overlay, (b["x0"], b["y0"]), (b["x1"], b["y1"]),
                                         color, sel)
            self.overlay = cv2.putText(
                self.overlay, name, (b["x0"], b["y0"] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self.overlay = cv2.circle(self.overlay, (self.anchor["x"], self.anchor["y"]), 6, (0, 0, 255), 2)

    def redraw(self):
        self.refresh_overlay()
        ok, buf = cv2.imencode(".png", self.overlay)
        if not ok:
            return
        img = tk.PhotoImage(data=buf.tobytes())
        self.canvas.img = img
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=img, anchor="nw")
        self.status.config(
            text=f"{FRAME_NAME} scale={self.scale:.2f} anchor=({self.anchor['x']},{self.anchor['y']}) | "
                 f"当前: {self.current} {self.boxes[self.current]}")
        self.root.after(30, self.redraw)

    def on_press(self, event):
        for name in REGIONS:
            b = self.boxes[name]
            if b["x0"] - 8 <= event.x <= b["x1"] + 8 and b["y0"] - 8 <= event.y <= b["y1"] + 8:
                self.current = name
                self.cursor[name] = (event.x - b["x0"], event.y - b["y0"])
                return

    def on_drag(self, event):
        for name in REGIONS:
            c = self.cursor[name]
            if c is not None:
                b = self.boxes[name]
                w = b["x1"] - b["x0"]
                h = b["y1"] - b["y0"]
                b["x0"] = event.x - c[0]
                b["y0"] = event.y - c[1]
                b["x1"] = b["x0"] + w
                b["y1"] = b["y0"] + h
                return

    def on_release(self, event):
        self.cursor = {k: None for k in self.cursor}

    def on_key(self, event):
        if event.char in "1234":
            self.current = REGIONS[int(event.char) - 1]
        elif event.char == "r":
            for name in REGIONS:
                self.boxes[name] = hud_regions.rel_to_abs(self.cfg["regions"][name], self.anchor)
        elif event.char == "s":
            self.save()
        elif event.char == "q":
            self.root.destroy()

    def save(self):
        data = {
            "video": "BV1Uu8z6eEVM",
            "anchor_frame": FRAME_NAME,
            "anchor": self.anchor,
            "regions": {
                name: hud_regions.abs_to_rel(
                    (b["x0"], b["y0"], b["x1"], b["y1"]), self.anchor)
                for name, b in self.boxes.items()
            },
        }
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.status.config(text=f"已保存到 {OUT}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

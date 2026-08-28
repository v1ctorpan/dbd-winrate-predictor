import json

REF_W, REF_H = 35, 32

def rel_to_abs(rel, anchor):
    s = anchor["scale"]
    ax, ay = anchor["x"], anchor["y"]
    return {
        "x0": int(round(ax + rel["x0"] * s)),
        "y0": int(round(ay + rel["y0"] * s)),
        "x1": int(round(ax + rel["x1"] * s)),
        "y1": int(round(ay + rel["y1"] * s)),
    }

def abs_to_rel(box, anchor):
    s = anchor["scale"]
    ax, ay = anchor["x"], anchor["y"]
    return {
        "x0": (box[0] - ax) / s,
        "y0": (box[1] - ay) / s,
        "x1": (box[2] - ax) / s,
        "y1": (box[3] - ay) / s,
    }

def load_regions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_regions(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def resolve_regions(config, anchor):
    resolved = {}
    for name, rel in config["regions"].items():
        resolved[name] = rel_to_abs(rel, anchor)
    return resolved

import os
import numpy as np
from PIL import Image
from src.backgrounds import make_paper, make_palm_leaf, add_margin_rules
from src.layout import layout_page
from src.effects import finish_page

kalam = "fonts/Kalam-Regular.ttf"
FONTS = {
    "devanagari": kalam if os.path.exists(kalam) else "fonts/NotoSansDevanagari-Regular.ttf",
    "modi": "fonts/NotoSansModi-Regular.ttf",
    "sharada": "fonts/NotoSansSharada-Regular.ttf",
}


def load(script):
    with open(f"data/texts/{script}_clean.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


for script, kind in (("devanagari", "paper"), ("modi", "palm"), ("sharada", "paper")):
    corpus = load(script)
    rng = np.random.default_rng(11)
    if kind == "paper":
        res = add_margin_rules(make_paper(1600, 720, rng), rng, "right")
    else:
        res = make_palm_leaf(2000, 450, rng)
    out = layout_page(res, FONTS[script], corpus, int(rng.integers(len(corpus))), rng, script)
    fin = finish_page(res, out["layer"], rng)
    outside = int(((fin["alpha"] > 0.1) & (fin["safe_mask"] < 0.5)).sum())
    print(f"{script}/{kind}: text pixels outside warped safe box = {outside}")
    Image.fromarray(fin["image"]).save(f"output/effects_{script}_{kind}.png")
    Image.fromarray((fin["alpha"] * 255).astype(np.uint8)).save(f"output/effects_{script}_{kind}_mask.png")
print("saved output/effects_*")
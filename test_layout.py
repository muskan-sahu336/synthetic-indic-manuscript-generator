import os
import numpy as np
from PIL import Image
from src.backgrounds import make_paper, make_palm_leaf, add_margin_rules
from src.renderer import composite
from src.layout import layout_page, make_marker

kalam = "fonts/Kalam-Regular.ttf"
FONTS = {
    "devanagari": kalam if os.path.exists(kalam) else "fonts/NotoSansDevanagari-Regular.ttf",
    "modi": "fonts/NotoSansModi-Regular.ttf",
    "sharada": "fonts/NotoSansSharada-Regular.ttf",
}


def load(script):
    with open(f"data/texts/{script}_clean.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


for script in ("devanagari", "modi", "sharada"):
    font = FONTS[script]
    corpus = load(script)
    print(f"\n=== {script} | font {font} | marker glyphs available: "
          f"{make_marker(script, 12, font) is not None}")
    for kind in ("paper", "palm"):
        rng = np.random.default_rng(7)
        if kind == "paper":
            res = add_margin_rules(make_paper(1600, 720, rng), rng, "right")
        else:
            res = make_palm_leaf(2000, 450, rng)
        out = layout_page(res, font, corpus, int(rng.integers(len(corpus))), rng, script)
        print(f"-- {kind}: {len(out['lines'])} lines, px={out['px']:.0f}, red={out['n_red']}, "
              f"dropped={out['dropped']}, clipped={out['layer'].clipped}, notes={out['notes']}")
        for ln in out["lines"]:
            print("   ", ln)
        Image.fromarray(composite(res["image"], out["layer"])).save(f"output/layout_{script}_{kind}.png")
print("\nsaved output/layout_<script>_<paper|palm>.png")
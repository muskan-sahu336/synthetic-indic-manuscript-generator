import os
import numpy as np
from src.backgrounds import make_paper, make_palm_leaf, add_margin_rules
from src.layout import layout_page
from src.effects import finish_page
from src.export import check_overflow, save_sample

kalam = "fonts/Kalam-Regular.ttf"
FONTS = {
    "devanagari": kalam if os.path.exists(kalam) else "fonts/NotoSansDevanagari-Regular.ttf",
    "modi": "fonts/NotoSansModi-Regular.ttf",
    "sharada": "fonts/NotoSansSharada-Regular.ttf",
}


def load(script):
    with open(f"data/texts/{script}_clean.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


for i, (script, kind) in enumerate((("devanagari", "paper"), ("modi", "palm"), ("sharada", "paper"))):
    corpus = load(script)
    rng = np.random.default_rng(11)
    if kind == "paper":
        res = add_margin_rules(make_paper(1600, 720, rng), rng, "right")
    else:
        res = make_palm_leaf(2000, 450, rng)
    out = layout_page(res, FONTS[script], corpus, int(rng.integers(len(corpus))), rng, script)
    fin = finish_page(res, out["layer"], rng, extra_boxes=out["regions"]["notes"])
    ok, n_out, n_edge = check_overflow(fin)
    print(f"{script}/{kind}: overflow ok={ok} outside={n_out} on_edge={n_edge} notes={len(out['notes'])}")
    img_p, md_p = save_sample(f"output/export_test/{script}", f"{script}_{i:03d}",
                              fin["image"], out["lines"], out["notes"])
    print("  saved", img_p, "and", md_p)
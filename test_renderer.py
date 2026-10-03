import os
import sys
import numpy as np
from PIL import Image
from src.backgrounds import make_paper, add_margin_rules
from src.renderer import TextLayer, draw_line, composite

candidates = [sys.argv[1]] if len(sys.argv) > 1 else [
    "fonts/Kalam-Regular.ttf", "fonts/NotoSansDevanagari-Regular.ttf"]
font = next(p for p in candidates if os.path.exists(p))
print("font:", font)

rng = np.random.default_rng(1)
res = make_paper(1600, 720, rng)
res = add_margin_rules(res, rng, "right")
left, top, right, bottom = res["safe_box"]
print("safe_box:", res["safe_box"])

with open("data/texts/devanagari_clean.txt", encoding="utf-8") as f:
    lines = [next(f).strip() for _ in range(7)]

layer = TextLayer(720, 1600)
px, pitch = 52, 78
for i, line in enumerate(lines):
    words = line.split()
    red = set(rng.choice(min(len(words), 5), size=2, replace=False)) if i % 3 == 1 else set()
    x_end, n = draw_line(layer, font, words, left, top + px + i * pitch, px, rng,
                        red=red, max_x=right)
    print(f"line {i}: placed {n}/{len(words)} words, ends at x={x_end:.0f}")

print("clipped by image border:", layer.clipped)
Image.fromarray(composite(res["image"], layer)).save("output/renderer_test.png")
Image.fromarray((layer.alpha * 255).astype(np.uint8)).save("output/renderer_mask.png")
print("saved output/renderer_test.png and output/renderer_mask.png")
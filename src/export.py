import os
import numpy as np
from PIL import Image


def check_overflow(fin, thresh=0.15, border=6):
    """No text pixels may fall outside the (warped) allowed regions or touch the page edge.
    Returns (ok, n_outside, n_on_edge)."""
    a = fin["alpha"] > thresh
    outside = int((a & (fin["safe_mask"] < 0.5)).sum())
    edge = np.zeros(a.shape, bool)
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True
    on_edge = int((a & edge).sum())
    return outside == 0 and on_edge == 0, outside, on_edge


def build_md(lines, notes=(), include_notes=True):
    """Plain text, one line per rendered line in reading order, no markup.
    Marginal notes (if any) go last, one per line."""
    out = [l for l in lines if l.strip()]
    if include_notes:
        out += [n for n in notes if n.strip()]
    return "\n".join(out) + "\n"


def save_sample(out_dir, stem, image, lines, notes=(), include_notes=True):
    """Write <stem>.png and <stem>.md. Reads the .md back to confirm it matches."""
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, stem + ".png")
    md_path = os.path.join(out_dir, stem + ".md")
    Image.fromarray(image).save(img_path)
    text = build_md(lines, notes, include_notes)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(md_path, encoding="utf-8", newline="") as f:
        assert f.read() == text, "md file does not match what was rendered"
    return img_path, md_path
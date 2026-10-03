import numpy as np
from fontTools.ttLib import TTFont
from .renderer import TextLayer, render_word, pick_ink, STYLE

_DIGIT_BASE = {"devanagari": 0x0966, "modi": 0x11650, "sharada": 0x111D0}
_CMAPS = {}

def register_digit_base(script, base):
    _DIGIT_BASE[script] = base

def _cmap(font_path):
    if font_path not in _CMAPS:
        _CMAPS[font_path] = TTFont(font_path, lazy=True).getBestCmap()
    return _CMAPS[font_path]


def make_marker(script, n, font_path):
    """Verse marker like ।१२। (danda + digits + danda). Returns None if the font lacks a glyph."""
    cm = _cmap(font_path)
    base = _DIGIT_BASE[script]
    s = "\u0964" + "".join(chr(base + int(d)) for d in str(n)) + "\u0964"
    return s if all(ord(c) in cm for c in s) else None


class _Stream:
    """Pulls words from consecutive corpus lines, inserting verse markers."""

    def __init__(self, corpus, start, marker_fn, every):
        self.corpus, self.i = corpus, start
        self.marker_fn, self.every = marker_fn, every
        self.buf, self.read = [], 0

    def next(self):
        while not self.buf:
            line = self.corpus[self.i % len(self.corpus)]
            self.i += 1
            self.read += 1
            self.buf = [(t, False) for t in line.split()]
            if self.buf and self.marker_fn and self.read % self.every == 0:
                m = self.marker_fn()
                if m:
                    self.buf.append((m, True))
        return self.buf.pop(0)


def _free_intervals(left, right, keepouts, y0, y1, min_w):
    segs = [(left, right)]
    for kl, kt, kr, kb in keepouts:
        if kb <= y0 or kt >= y1:
            continue
        new = []
        for a, b in segs:
            if kr <= a or kl >= b:
                new.append((a, b))
                continue
            if kl > a:
                new.append((a, kl))
            if kr < b:
                new.append((kr, b))
        segs = new
    return [(a, b) for a, b in segs if b - a >= min_w]


def _blit(layer, alpha, color, x0, y0):
    H, W = layer.alpha.shape
    h, w = alpha.shape
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        layer.clipped = True
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, W), min(y0 + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    c = color[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    A = layer.alpha[cy0:cy1, cx0:cx1]
    new = A + a - A * a
    wgt = (a / np.maximum(new, 1e-6))[..., None]
    layer.color[cy0:cy1, cx0:cx1] = layer.color[cy0:cy1, cx0:cx1] * (1 - wgt) + c * wgt
    layer.alpha[cy0:cy1, cx0:cx1] = new


def layout_page(res, font_path, corpus, start, rng, script="devanagari",
                n_lines=None, notes=True, style=STYLE):
    """Lay out one page. Returns a dict:
       layer      TextLayer (text mask + colours)
       lines      list[str], main text exactly as placed, one string per visual line
       notes      list[str], marginal note texts (may be empty)
       next_idx   corpus index to continue from
       regions    {"main": safe_box, "notes": [boxes]}
       dropped    number of words skipped because they were wider than any line
    """
    H, W = res["image"].shape[:2]
    left, top, right, bottom = res["safe_box"]
    keepouts = res["keepouts"]
    kind = res["kind"]

    if n_lines is None:
        n_lines = int(rng.integers(7, 11)) if kind == "paper" else int(rng.integers(3, 6))
    slack = 12
    box_h = bottom - top - 2 * slack
    px = float(np.clip(box_h / (n_lines * 1.4), 34, 56))
    pitch = (box_h - 1.3 * px) / max(n_lines - 1, 1)
    first = top + slack + 0.95 * px

    slant = rng.uniform(*style["slant"])
    page_t = int(rng.integers(0, 2))

    marker_n = [int(rng.integers(1, 40))]

    def marker_fn():
        m = make_marker(script, marker_n[0], font_path)
        marker_n[0] += 1
        return m

    stream = _Stream(corpus, start, marker_fn, int(rng.choice([1, 2])))
    dropped = [0]

    segs0 = _free_intervals(left, right, keepouts, first - px, first + 0.4 * px, 3 * px)
    max_w = max([b - a for a, b in segs0] or [right - left])

    def fetch():
        while True:
            tok, is_m = stream.next()
            t = page_t + (1 if rng.random() < 0.15 else 0)
            wd = render_word(font_path, tok, px, rng, slant, style, thick=t)
            if wd is None:
                continue
            if wd["alpha"].shape[1] > max_w:
                dropped[0] += 1
                continue
            return tok, is_m, wd

    # ---- pass 1: wrap and position words ----
    lines, flat, pending = [], [], None
    for li in range(n_lines):
        base0 = first + li * pitch + rng.normal(0, 1.0)
        rl = right - rng.uniform(0, 0.04) * (right - left)
        segs = _free_intervals(left, rl, keepouts, base0 - px, base0 + 0.4 * px, 3 * px)
        amp = rng.uniform(*style["wave_amp"])
        wl = rng.uniform(*style["wave_len"])
        ph = rng.uniform(0, 2 * np.pi)
        total_drift = rng.uniform(-7, 7)
        off = rng.uniform(0, 0.012 * W)
        x_ref = segs[0][0] if segs else left
        row = []
        for si, (x0, x1) in enumerate(segs):
            cursor = None
            while True:
                if pending is None:
                    pending = fetch()
                tok, is_m, wd = pending
                ax = wd["ax"]
                ext = wd["alpha"].shape[1] - ax
                origin = (x0 + (off if si == 0 else 0) + ax) if cursor is None else cursor
                if origin + ext > x1:
                    break
                dx = origin - x_ref
                by = (base0 + amp * np.sin(2 * np.pi * dx / wl + ph)
                      + total_drift * dx / max(right - left, 1)
                    + rng.normal(0, style["jit_y"]))
                rec = {"tok": tok, "marker": is_m, "wd": wd, "x": origin, "y": by, "line": li}
                row.append(rec)
                flat.append(rec)
                pending = None
                cursor = origin + wd["adv"] + px * rng.uniform(*style["gap"])
        lines.append(row)

    # ---- pick red (rubricated) words from what was actually placed ----
    cand = [i for i, r in enumerate(flat) if not r["marker"]]
    n_red = int(rng.integers(3, 8)) if kind == "paper" else int(rng.integers(2, 6))
    red = set()
    if cand:
        for i in rng.choice(cand, size=min(n_red, len(cand)), replace=False):
            i = int(i)
            red.add(i)
            j = i + 1
            if (rng.random() < 0.45 and j < len(flat) and not flat[j]["marker"]
                    and flat[j]["line"] == flat[i]["line"]):
                red.add(j)

    # ---- pass 2: draw ----
    layer = TextLayer(H, W)
    line_ink = {}
    for i, r in enumerate(flat):
        if r["line"] not in line_ink:
            line_ink[r["line"]] = pick_ink(rng)
        col = pick_ink(rng, True) if i in red else line_ink[r["line"]] + rng.normal(0, 3, 3)
        layer.place(r["wd"], r["x"], r["y"], col, fade=rng.uniform(*style["fade"]))

    main_lines = [" ".join(r["tok"] for r in row) for row in lines if row]

    # ---- optional marginal note: small rotated text in the left margin ----
    note_texts, note_boxes = [], []
    if notes and kind == "paper" and rng.random() < 0.6:
        pxn = float(rng.uniform(26, 32))
        hn = int(1.35 * pxn) + 6
        if left - 12 >= hn:
            max_len = 0.55 * (bottom - top)
            target = int(rng.integers(2, 5))
            words, total, tries = [], 0.0, 0
            while len(words) < target and tries < 12:
                tries += 1
                tok, is_m = stream.next()
                if is_m:
                    continue
                wd = render_word(font_path, tok, pxn, rng, slant, style, thick=0)
                if wd is None:
                    continue
                need = wd["adv"] + pxn * 0.3
                if total + need > max_len:
                    continue
                words.append((tok, wd))
                total += need
            if words:
                Wn = int(total + pxn) + 4
                sl = TextLayer(hn, Wn)
                cur = 2.0
                for tok, wd in words:
                    sl.place(wd, cur + wd["ax"], 0.95 * pxn + 3, [0, 0, 0])
                    cur += wd["adv"] + pxn * 0.3
                col = pick_ink(rng, red=rng.random() < 0.25)
                A = np.rot90(sl.alpha) * 0.9                       # 90 deg counter-clockwise
                C = np.broadcast_to(col, (A.shape[0], A.shape[1], 3)).astype(np.float32)
                cx = left * 0.5 + rng.uniform(-4, 4)
                x0 = int(np.clip(cx - hn / 2, 6, left - hn - 6))
                y0 = int(top + rng.uniform(0, max(1.0, (bottom - top) - A.shape[0])))
                _blit(layer, A.astype(np.float32), C, x0, y0)
                note_texts.append(" ".join(t for t, _ in words))
                note_boxes.append((x0, y0, x0 + hn, y0 + A.shape[0]))

    return {"layer": layer, "lines": main_lines, "notes": note_texts,
            "next_idx": stream.i, "regions": {"main": res["safe_box"], "notes": note_boxes},
            "dropped": dropped[0], "px": px, "n_red": len(red)}
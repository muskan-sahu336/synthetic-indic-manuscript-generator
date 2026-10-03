import numpy as np
import cv2


def _noise(h, w, scale, rng):
    """Smooth random field with features about `scale` px wide."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def _fractal(h, w, rng, scales, decay=0.55):
    out, amp = np.zeros((h, w), np.float32), 1.0
    for s in scales:
        out += amp * _noise(h, w, s, rng)
        amp *= decay
    return _norm(out)


def _lighting(h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    g = np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5)
    return 1 + 0.12 * g


def _edge_darkness(h, w, rng, width):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    ragged = _fractal(h, w, rng, (width, width // 2, width // 4)) * (width * 0.3)
    return np.clip(1 - (d + ragged - 6) / width, 0, 1)


def _tint(img, mask, color, strength):
    a = (strength * mask)[..., None]
    return img * (1 - a) + a * np.array(color, np.float32)


PAPER_TONES = [(214, 184, 132), (204, 172, 118), (222, 196, 146),
            (192, 160, 108), (210, 180, 135)]
LEAF_TONES = [(176, 146, 88), (166, 140, 86), (158, 136, 84),
            (188, 160, 104), (164, 146, 90)]

def make_paper(w=1600, h=720, rng=None):
    rng = rng or np.random.default_rng()
    base = np.array(PAPER_TONES[rng.integers(len(PAPER_TONES))], np.float32)
    base = base + rng.normal(0, 6, 3)
    img = np.ones((h, w, 3), np.float32) * base

    # broad tonal patchiness
    img *= (0.74 + 0.38 * _fractal(h, w, rng, (300, 150, 80, 40)))[..., None]

    # soft stains (kept weak; the tide line below gives the crisp look)
    stain = np.clip((_fractal(h, w, rng, (200, 100, 50)) - rng.uniform(0.55, 0.7)) * 4, 0, 1)
    img = _tint(img, stain, (140, 95, 50), rng.uniform(0.15, 0.3))

    # water stain with a thin darker rim, limited to one or two regions
    f = _fractal(h, w, rng, (260, 120, 60))
    t = rng.uniform(0.55, 0.65)
    region = np.clip((_fractal(h, w, rng, (500, 250)) - rng.uniform(0.45, 0.6)) * 5, 0, 1)
    img = _tint(img, np.clip((f - t) * 6, 0, 1) * region, (150, 105, 55), rng.uniform(0.12, 0.22))
    img = _tint(img, np.exp(-((f - t) / 0.015) ** 2) * region, (115, 75, 40), rng.uniform(0.2, 0.35))

    # fibres
    fib = np.zeros((h, w), np.float32)
    for _ in range(int(w * h / 1800)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        L, ang = rng.integers(10, 40), rng.uniform(0, np.pi)
        cv2.line(fib, (int(x), int(y)),
                 (int(x + L * np.cos(ang)), int(y + L * np.sin(ang))),
                float(rng.uniform(0.3, 1)), 1)
    fib = cv2.GaussianBlur(fib, (0, 0), 0.8)
    img *= (1 - 0.18 * fib)[..., None]

    # foxing spots: irregular ellipses of varying size and strength
    spots = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(60, 200))):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ax = (int(rng.integers(1, 9)), int(rng.integers(1, 7)))
        cv2.ellipse(spots, c, ax, float(rng.uniform(0, 180)), 0, 360,
                    float(rng.uniform(0.3, 1.0)), -1)
    spots = cv2.GaussianBlur(spots, (0, 0), float(rng.uniform(1.0, 2.5)))
    spots *= 0.6 + 0.8 * _fractal(h, w, rng, (12, 6))
    img = _tint(img, np.clip(spots, 0, 1), (150, 100, 55), 0.5)

    # fine-scale mottling, grain, lighting, edges
    img *= (0.93 + 0.12 * _fractal(h, w, rng, (30, 15, 8)))[..., None]
    img += rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    img *= _lighting(h, w, rng)[..., None]
    img *= (1 - 0.35 * _edge_darkness(h, w, rng, 90) ** 2)[..., None]

    # margins measured from the official sample: wide bottom, moderate top
    mx = int(w * rng.uniform(0.07, 0.09))
    mt = int(h * rng.uniform(0.09, 0.11))
    mb = int(h * rng.uniform(0.12, 0.15))
    return {"kind": "paper",
        "image": np.clip(img, 0, 255).astype(np.uint8),
        "safe_box": (mx, mt, w - mx, h - mb),
        "keepouts": []}

def _streaks(h, w, rng):
    out = np.zeros((h, w), np.float32)
    for gh, gw, a in ((h // 2, w // 150, 1.0), (h, w // 60, 0.6), (h, w // 25, 0.4)):
        g = rng.random((max(2, gh), max(2, gw))).astype(np.float32)
        out += a * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return _norm(out)


def make_palm_leaf(w=2000, h=450, rng=None):
    rng = rng or np.random.default_rng()
    base = np.array(LEAF_TONES[rng.integers(len(LEAF_TONES))], np.float32)
    base = base + rng.normal(0, 6, 3)
    img = np.ones((h, w, 3), np.float32) * base

    # fine parallel streaks (calmer) + broad tonal patches
    img *= (0.88 + 0.22 * _streaks(h, w, rng))[..., None]
    img *= (0.85 + 0.30 * _fractal(h, w, rng, (500, 250)))[..., None]

    green = _fractal(h, w, rng, (400, 200, 100))
    img = _tint(img, np.clip((green - 0.5) * 2, 0, 1), (150, 160, 95), rng.uniform(0.1, 0.35))

    # veins and small cracks from the top/bottom edges
    veins = np.zeros((h, w), np.float32)
    for _ in range(60):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        cv2.line(veins, (x, y), (x + int(rng.integers(200, 1200)), y + int(rng.integers(-3, 4))),
                float(rng.uniform(0.3, 1)), 1)
    for _ in range(int(rng.integers(8, 25))):
        x, L = int(rng.integers(0, w)), int(rng.integers(10, 45))
        top = rng.random() < 0.5
        y0 = 0 if top else h - 1
        cv2.line(veins, (x, y0), (x + int(rng.integers(-4, 5)), y0 + (L if top else -L)), 1.0, 1)
    veins = cv2.GaussianBlur(veins, (0, 0), 1.0)
    img *= (1 - 0.28 * veins)[..., None]

    img = _tint(img, np.clip((_fractal(h, w, rng, (150, 70, 30)) - 0.65) * 4, 0, 1),
                (120, 85, 45), rng.uniform(0.2, 0.45))

    img *= (0.93 + 0.12 * _fractal(h, w, rng, (30, 15, 8)))[..., None]
    img += rng.normal(0, 3, (h, w, 1)).astype(np.float32)
    img *= _lighting(h, w, rng)[..., None]
    img *= (1 - 0.45 * _edge_darkness(h, w, rng, 55) ** 2)[..., None]

    # string holes: soft shadow ring, irregular dark hole, lighter rim
    keepouts, cy = [], h // 2
    n_holes = int(rng.integers(1, 3))
    xs = [int(w * rng.uniform(0.22, 0.32)), int(w * rng.uniform(0.68, 0.78))][:n_holes] \
        if n_holes == 2 else [int(w * rng.uniform(0.35, 0.65))]
    for cx in xs:
        r = int(rng.integers(11, 17))
        ry = int(r * rng.uniform(0.8, 1.1))
        ang = float(rng.uniform(0, 180))

        ring = np.zeros((h, w), np.float32)
        cv2.circle(ring, (cx, cy), r + 8, 1.0, -1)
        ring = cv2.GaussianBlur(ring, (0, 0), 5)
        img *= (1 - 0.35 * ring)[..., None]

        rim = np.zeros((h, w), np.float32)
        cv2.ellipse(rim, (cx, cy), (r + 3, ry + 3), ang, 0, 360, 1.0, 1)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.5)
        img = _tint(img, np.clip(rim * 2, 0, 1), (205, 180, 125), 0.45)

        hole = np.zeros((h, w), np.float32)
        cv2.ellipse(hole, (cx, cy), (r, ry), ang, 0, 360, 1.0, -1)
        hole = cv2.GaussianBlur(hole, (0, 0), 1.2)
        img = _tint(img, hole, (45, 35, 25), 0.95)

        keepouts.append((cx - r - 40, 0, cx + r + 40, h))

    mx, my = int(w * 0.04), int(h * rng.uniform(0.10, 0.14))
    return {"kind": "palm",
            "image": np.clip(img, 0, 255).astype(np.uint8),
            "safe_box": (mx, my, w - mx, h - my),
            "keepouts": keepouts}


def add_margin_rules(res, rng, side="right"):
    """Draw double red vertical rules near one edge and pull safe_box clear of them.
    Modifies and returns `res`. Skip this function if you already have your own."""
    img = res["image"].astype(np.float32)
    h, w = img.shape[:2]
    left, top, right, bottom = res["safe_box"]

    off = int(w * rng.uniform(0.11, 0.13))
    gap = int(w * rng.uniform(0.009, 0.013))
    pad = int(w * 0.09)                       # clearance between text and rules
    if side == "right":
        xs = [w - off, w - off - gap]
    else:
        xs = [off, off + gap]

    xx = np.arange(w, dtype=np.float32)[None, :]
    for xc in xs:
        # rule wobbles slightly along its length and fades in patches
        wob = np.interp(np.arange(h), np.linspace(0, h, 12), rng.normal(0, 1.2, 12))
        d = xx - (xc + wob)[:, None]
        sigma = float(rng.uniform(1.1, 1.6))
        mask = np.exp(-(d / sigma) ** 2).astype(np.float32)
        fade = 0.65 + 0.35 * _fractal(h, w, rng, (60, 25))
        img = _tint(img, mask * fade, (170, 55, 40), 0.8)

    if side == "right":
        res["safe_box"] = (left, top, min(right, xs[1] - pad), bottom)
    else:
        res["safe_box"] = (max(left, xs[1] + pad), top, right, bottom)
    res["image"] = np.clip(img, 0, 255).astype(np.uint8)
    return res
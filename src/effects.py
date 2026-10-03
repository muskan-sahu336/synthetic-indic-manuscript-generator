import numpy as np
import cv2
from .backgrounds import _fractal
from .renderer import composite

EFFECTS = {
    "bleed_sigma": (1.0, 2.0),
    "bleed_strength": (0.25, 0.45),
    "fade_strength": (0.25, 0.5),
    "dropout": 0.55,
    "n_smudges": (2, 5),
    "n_blots": (1, 4),
    "warp_amp": (4.0, 9.0),        # px of smooth displacement
    "curl_max": (6.0, 16.0),       # px pulled inward at the right edge
    "shade": (0.10, 0.18),         # wrinkle shading strength
    "n_folds": (0, 3),             # paper only
}


def _smudge(a, pre, rng, ys, xs):
    H, W = a.shape
    i = int(rng.integers(len(ys)))
    cy, cx = int(ys[i]), int(xs[i])
    r = int(rng.integers(25, 70))
    y0, y1 = max(cy - r, 0), min(cy + r, H)
    x0, x1 = max(cx - r, 0), min(cx + r, W)
    L = int(rng.integers(4, 10)) * 2 + 1
    ang = rng.uniform(-0.5, 0.5) if rng.random() < 0.7 else rng.uniform(0, np.pi)
    c = L // 2
    k = np.zeros((L, L), np.float32)
    cv2.line(k, (int(round(c - c * np.cos(ang))), int(round(c - c * np.sin(ang)))),
             (int(round(c + c * np.cos(ang))), int(round(c + c * np.sin(ang)))), 1.0, 1)
    k /= k.sum()
    ra = np.ascontiguousarray(a[y0:y1, x0:x1])
    rp = np.ascontiguousarray(pre[y0:y1, x0:x1])
    ba = cv2.filter2D(ra, -1, k)
    bp = cv2.filter2D(rp, -1, k)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2.2) ** 2)))
    m *= rng.uniform(0.25, 0.55)
    a[y0:y1, x0:x1] = ra * (1 - m) + ba * m
    pre[y0:y1, x0:x1] = rp * (1 - m[..., None]) + bp * m[..., None]


def _blot(a, pre, rng, ys, xs):
    H, W = a.shape
    i = int(rng.integers(len(ys)))
    cy, cx = int(ys[i]), int(xs[i])
    R = 16
    y0, y1 = max(cy - R, 0), min(cy + R, H)
    x0, x1 = max(cx - R, 0), min(cx + R, W)
    b = np.zeros((y1 - y0, x1 - x0), np.float32)
    for _ in range(int(rng.integers(2, 4))):
        cv2.ellipse(b, (cx - x0 + int(rng.integers(-4, 5)), cy - y0 + int(rng.integers(-3, 4))),
                    (int(rng.integers(3, 9)), int(rng.integers(3, 8))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    b = cv2.GaussianBlur(b, (0, 0), 1.3) * rng.uniform(0.6, 0.95)
    c = pre[cy, cx] / max(float(a[cy, cx]), 1e-6)
    ra = a[y0:y1, x0:x1].copy()
    add = b * (1 - ra)
    a[y0:y1, x0:x1] = ra + add
    pre[y0:y1, x0:x1] += add[..., None] * c


def apply_ink_effects(layer, rng, cfg=EFFECTS):
    """Bleed, fading, speckle dropout, smudges and blots. Modifies layer in place."""
    a = layer.alpha.copy()
    H, W = a.shape
    if a.max() <= 0:
        return layer
    pre = layer.color * a[..., None]

    # bleed: ink soaks outward a little, unevenly
    sig = float(rng.uniform(*cfg["bleed_sigma"]))
    ba = cv2.GaussianBlur(a, (0, 0), sig)
    bp = cv2.GaussianBlur(pre, (0, 0), sig)
    absorb = 0.6 + 0.8 * _fractal(H, W, rng, (6, 3))
    bleed = np.clip(ba * rng.uniform(*cfg["bleed_strength"]) * absorb, 0, 1) * (1 - a)
    bcol = bp / np.maximum(ba[..., None], 1e-6)
    a = a + bleed
    pre = pre + bleed[..., None] * bcol

    # faded patches (low-frequency) and speckle dropout (high-frequency)
    patch = np.clip((_fractal(H, W, rng, (260, 120)) - rng.uniform(0.55, 0.7)) * 4, 0, 1)
    fac = 1 - rng.uniform(*cfg["fade_strength"]) * patch
    speck = np.clip((_fractal(H, W, rng, (3, 2)) - 0.72) * 6, 0, 1)
    fac = fac * (1 - cfg["dropout"] * speck)
    a *= fac
    pre *= fac[..., None]

    # smudges and blots, positioned on text
    ys, xs = np.nonzero(a > 0.5)
    if len(ys):
        for _ in range(int(rng.integers(*cfg["n_smudges"]))):
            _smudge(a, pre, rng, ys, xs)
        for _ in range(int(rng.integers(*cfg["n_blots"]))):
            _blot(a, pre, rng, ys, xs)

    a = np.clip(a, 0, 1)
    layer.alpha = a
    layer.color = pre / np.maximum(a[..., None], 1e-6)
    return layer

def warp_page(image, alpha, safe_boxes, rng, kind="paper", cfg=EFFECTS):
    """Curl, wrinkle and fold the page. The same displacement is applied to the image,
    the text mask and a safe-box mask, so all three stay aligned.
    Returns dict: image (uint8), alpha (float32), safe_mask (float32)."""
    H, W = alpha.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    amp = float(rng.uniform(*cfg["warp_amp"]))
    dx = (_fractal(H, W, rng, (420, 200)) - 0.5) * 2 * amp
    dy = (_fractal(H, W, rng, (420, 200)) - 0.5) * 2 * amp

    # page curl: right edge pulls inward (content shifts left)
    cw = W * rng.uniform(0.05, 0.10)
    ramp = np.clip((xx - (W - cw)) / cw, 0, 1) ** 2
    dx = dx + rng.uniform(*cfg["curl_max"]) * ramp

    map_x, map_y = xx + dx, yy + dy

    def rm(src, border):
        return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=border, borderValue=0)

    img = rm(image.astype(np.float32), cv2.BORDER_REFLECT)
    al = rm(alpha.astype(np.float32), cv2.BORDER_CONSTANT)
    sm = np.zeros((H, W), np.float32)
    sm = np.zeros((H, W), np.float32)
    for l, t, r, b in safe_boxes:
        sm[t:b, l:r] = 1.0
    sm = rm(sm, cv2.BORDER_CONSTANT)

    # wrinkle shading from a smooth height field, lit from the upper left
    hf = cv2.GaussianBlur(_fractal(H, W, rng, (160, 80, 40)), (0, 0), 2.0)
    g = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) * 0.7 + cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) * 0.7
    g = np.clip(g / (g.std() * 3 + 1e-6), -1, 1)
    shade = 1 + rng.uniform(*cfg["shade"]) * g
    shade *= 1 - 0.18 * ramp                                   # curled edge is darker

    # fold lines (paper only)
    if kind == "paper":
        for _ in range(int(rng.integers(*cfg["n_folds"]))):
            vertical = rng.random() < 0.6
            pos = rng.uniform(0.15, 0.85) * (W if vertical else H)
            tilt = rng.uniform(-0.03, 0.03)
            d = (xx - pos - tilt * yy) if vertical else (yy - pos - tilt * xx)
            crease = np.exp(-(d / 1.6) ** 2)
            glint = np.exp(-((d - 3) / 1.6) ** 2)
            shade *= (1 - 0.22 * rng.uniform(0.5, 1) * crease + 0.08 * glint) * (1 + 0.05 * np.tanh(d / 40))

    img = np.clip(img * shade[..., None], 0, 255).astype(np.uint8)
    return {"image": img, "alpha": np.clip(al, 0, 1), "safe_mask": sm}


def finish_page(res, layer, rng, extra_boxes=(), cfg=EFFECTS):
    """Ink effects -> composite -> warp. extra_boxes = allowed regions outside the main box (margin notes)."""
    apply_ink_effects(layer, rng, cfg)
    img = composite(res["image"], layer)
    boxes = [res["safe_box"]] + [(max(int(b[0]) - 8, 0), max(int(b[1]) - 8, 0),
                                int(b[2]) + 8, int(b[3]) + 8) for b in extra_boxes]
    return warp_page(img, layer.alpha, boxes, rng, res["kind"], cfg)
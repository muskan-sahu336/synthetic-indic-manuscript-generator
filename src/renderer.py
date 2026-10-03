import numpy as np
import cv2
import uharfbuzz as hb
import freetype

SS = 3  # supersampling factor

STYLE = {
    "slant": (0.05, 0.16),        # shear per line (tan of slant angle)
    "slant_jitter": 0.04,         # extra per-word variation
    "rot_deg": 1.5,               # max per-word rotation
    "size_jitter": (0.94, 1.07),
    "thick": (-1, 2),             # erode(-1) .. dilate(+2) steps at hi-res
    "fade": (0.80, 1.0),          # per-word ink strength
    "gap": (0.20, 0.38),          # word gap in em
    "wave_amp": (1.0, 3.5),       # baseline sine amplitude, px
    "wave_len": (250, 600),       # baseline sine wavelength, px
    "drift": (-6, 6),             # baseline drift over 1000 px
    "jit_y": 0.8,                 # random per-word vertical jitter, px
}

INKS = [(46, 34, 26), (54, 38, 29), (62, 44, 33), (72, 52, 39)]
REDS = [(172, 62, 40), (160, 55, 38), (184, 78, 48)]

_FACES = {}
_KERNEL = np.ones((3, 3), np.uint8)


def _load(font_path):
    if font_path not in _FACES:
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        _FACES[font_path] = (hb.Font(face), face.upem, freetype.Face(font_path), blob)
    return _FACES[font_path]


def _shape_hi(font_path, text, px_hi):
    """Shape one word and rasterise it at hi-res.
    Returns (alpha uint8, ox, oy, advance): (ox, oy) = baseline origin inside the array."""
    hb_font, upem, ft, _ = _load(font_path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {"kern": True})
    ft.set_pixel_sizes(0, px_hi)
    scale = px_hi / upem

    glyphs, x = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        g = ft.glyph
        bmp = g.bitmap
        if bmp.rows and bmp.width:
            arr = np.array(bmp.buffer, np.uint8).reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
            gx = int(round(x + pos.x_offset * scale)) + g.bitmap_left
            gy = -int(round(pos.y_offset * scale)) - g.bitmap_top   # relative to baseline, y down
            glyphs.append((gx, gy, arr))
        x += pos.x_advance * scale
    if not glyphs:
        return None

    pad = 2
    minx = min(g[0] for g in glyphs)
    maxx = max(g[0] + g[2].shape[1] for g in glyphs)
    miny = min(g[1] for g in glyphs)
    maxy = max(g[1] + g[2].shape[0] for g in glyphs)
    W, H = maxx - minx + 2 * pad, maxy - miny + 2 * pad
    ox, oy = pad - minx, pad - miny
    canvas = np.zeros((H, W), np.uint8)
    for gx, gy, arr in glyphs:
        y0, x0 = gy + oy, gx + ox
        sub = canvas[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]]
        np.maximum(sub, arr, out=sub)
    return canvas, ox, oy, x


def render_word(font_path, text, px, rng, slant, style=STYLE, thick=None):
    """Render one word with handwriting-like variation.
    Returns dict: alpha (float32 0..1), ax/ay (baseline origin in alpha), adv (advance in px)."""
    res = _shape_hi(font_path, text, int(round(px * SS)))
    if res is None:
        return None
    hi, ox, oy, adv_hi = res

    t = thick if thick is not None else int(rng.integers(style["thick"][0], style["thick"][1] + 1))
    if t > 0:
        hi = cv2.dilate(hi, _KERNEL, iterations=t)
    elif t < 0:
        hi = cv2.erode(hi, _KERNEL, iterations=-t)

    k = slant + rng.normal(0, style["slant_jitter"])
    th = np.deg2rad(rng.normal(0, style["rot_deg"] / 2))
    s = rng.uniform(*style["size_jitter"])
    shear = np.array([[1.0, -k], [0.0, 1.0]])      # top of glyph moves right
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = rot @ (s * np.eye(2)) @ shear

    h, w = hi.shape
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) - [ox, oy]
    pts = corners @ A.T
    mn, mx = pts.min(0), pts.max(0)
    out_w = int(np.ceil((mx[0] - mn[0]) / SS)) * SS
    out_h = int(np.ceil((mx[1] - mn[1]) / SS)) * SS
    anchor = -mn                                    # baseline origin in output (hi-res)
    M = np.hstack([A, (anchor - A @ np.array([ox, oy], np.float64))[:, None]])
    warped = cv2.warpAffine(hi, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    small = cv2.resize(warped, (out_w // SS, out_h // SS), interpolation=cv2.INTER_AREA)
    return {"alpha": small.astype(np.float32) / 255.0,
            "ax": anchor[0] / SS, "ay": anchor[1] / SS,
            "adv": adv_hi * s / SS}


class TextLayer:
    """Text-only layer: coverage mask + per-pixel ink colour."""

    def __init__(self, h, w):
        self.alpha = np.zeros((h, w), np.float32)
        self.color = np.zeros((h, w, 3), np.float32)
        self.clipped = False   # True if any word was cut off by the image border

    def place(self, word, x, y, color, fade=1.0):
        a = word["alpha"] * fade
        x0 = int(round(x - word["ax"]))
        y0 = int(round(y - word["ay"]))
        hh, ww = a.shape
        H, W = self.alpha.shape
        if x0 < 0 or y0 < 0 or x0 + ww > W or y0 + hh > H:
            self.clipped = True
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + ww, W), min(y0 + hh, H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        b = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        A_ = self.alpha[cy0:cy1, cx0:cx1]
        new = A_ + b - A_ * b
        wgt = (b / np.maximum(new, 1e-6))[..., None]
        col = self.color[cy0:cy1, cx0:cx1]
        self.color[cy0:cy1, cx0:cx1] = col * (1 - wgt) + np.array(color, np.float32) * wgt
        self.alpha[cy0:cy1, cx0:cx1] = new


def pick_ink(rng, red=False):
    base = REDS[rng.integers(len(REDS))] if red else INKS[rng.integers(len(INKS))]
    return np.clip(np.array(base, np.float32) + rng.normal(0, 4, 3), 0, 255)


def draw_line(layer, font_path, words, x, baseline, px, rng,
            red=(), max_x=None, style=STYLE):
    """Draw words left to right with a wavy baseline. Stops before a word that would pass max_x.
    Returns (x_end, n_words_placed)."""
    ink = pick_ink(rng)
    line_t = int(rng.integers(0, 2))   # base pen weight for this line
    slant = rng.uniform(*style["slant"])
    amp = rng.uniform(*style["wave_amp"])
    wl = rng.uniform(*style["wave_len"])
    ph = rng.uniform(0, 2 * np.pi)
    drift = rng.uniform(*style["drift"])
    x0, n = x, 0
    for i, text in enumerate(words):
        wd = render_word(font_path, text, px, rng, slant, style,
            thick=line_t + (1 if rng.random() < 0.15 else 0))
        if wd is None:
            continue
        if max_x is not None and x + wd["adv"] > max_x:
            break
        dx = x - x0
        by = (baseline + amp * np.sin(2 * np.pi * dx / wl + ph)
              + drift * dx / 1000.0 + rng.normal(0, style["jit_y"]))
        col = pick_ink(rng, True) if i in red else ink + rng.normal(0, 3, 3)
        layer.place(wd, x, by, col, fade=rng.uniform(*style["fade"]))
        x += wd["adv"] + px * rng.uniform(*style["gap"])
        n += 1
    return x, n


def composite(bg, layer):
    """Blend the text layer onto an RGB uint8 background; ink is tinted by the paper tone."""
    bgf = bg.astype(np.float32)
    a = layer.alpha[..., None]
    ink = layer.color * (0.75 + 0.25 * bgf / 255.0)
    return np.clip(bgf * (1 - a) + ink * a, 0, 255).astype(np.uint8)
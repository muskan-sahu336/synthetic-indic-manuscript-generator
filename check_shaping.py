import numpy as np
import uharfbuzz as hb
import freetype
from PIL import Image
from aksharamukha import transliterate


def render_line(font_path, text, px=56, width=900, height=100):
    hb_face = hb.Face(hb.Blob.from_file_path(font_path))
    hb_font = hb.Font(hb_face)
    scale = px / hb_face.upem

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {"kern": True})

    ft = freetype.Face(font_path)
    ft.set_pixel_sizes(0, px)

    canvas = np.zeros((height, width), dtype=np.uint8)
    x, baseline = 20.0, 70

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        g = ft.glyph
        bmp = g.bitmap
        if bmp.rows and bmp.width:
            arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
            gx = int(round(x + pos.x_offset * scale)) + g.bitmap_left
            gy = int(round(baseline - pos.y_offset * scale)) - g.bitmap_top
            y0, x0 = max(gy, 0), max(gx, 0)
            y1 = min(gy + bmp.rows, height)
            x1 = min(gx + bmp.width, width)
            if y1 > y0 and x1 > x0:
                sub = arr[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)
        x += pos.x_advance * scale

    return 255 - canvas  # black text on white


base = "क्षत्रिय ज्ञान श्री द्वारा"
rows = [
    render_line("fonts/NotoSansDevanagari-Regular.ttf", base),
    render_line("fonts/NotoSansModi-Regular.ttf",
                transliterate.process("Devanagari", "Modi", base)),
    render_line("fonts/NotoSansSharada-Regular.ttf",
                transliterate.process("Devanagari", "Sharada", base)),
]

Image.fromarray(np.vstack(rows)).save("output/shaping_test_hb.png")
print("Saved output/shaping_test_hb.png")
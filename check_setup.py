from PIL import Image, ImageDraw, ImageFont, features
from aksharamukha import transliterate

print("raqm available:", features.check("raqm"))

base = "क्षत्रिय ज्ञान श्री द्वारा"

tests = [
    ("fonts/NotoSansDevanagari-Regular.ttf", base),
    ("fonts/NotoSansModi-Regular.ttf",
     transliterate.process("Devanagari", "Modi", base)),
    ("fonts/NotoSansSharada-Regular.ttf",
     transliterate.process("Devanagari", "Sharada", base)),
]

img = Image.new("RGB", (900, 100 * len(tests)), "white")
d = ImageDraw.Draw(img)
for i, (path, text) in enumerate(tests):
    font = ImageFont.truetype(path, 56, layout_engine=ImageFont.Layout.RAQM)
    d.text((20, 15 + i * 100), text, font=font, fill="black")

img.save("output/shaping_test.png")
print("Saved output/shaping_test.png")

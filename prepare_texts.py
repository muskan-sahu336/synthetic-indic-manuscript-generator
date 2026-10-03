import re
from fontTools.ttLib import TTFont

SCRIPTS = {
    "devanagari": ("data/texts/raw_devanagari.md", "fonts/NotoSansDevanagari-Regular.ttf"),
    "modi":       ("data/texts/raw_modi.md",       "fonts/NotoSansModi-Regular.ttf"),
    "sharada":    ("data/texts/raw_sharada.md",    "fonts/NotoSansSharada-Regular.ttf"),
}
MIN_LEN, MAX_LEN = 20, 300
JOINERS = {"\u200c", "\u200d"}


def clean(line):
    line = line.replace("\u00a0", " ")
    if re.search(r"[A-Za-z0-9]", line):          # drop lines with Latin letters/digits
        return None
    line = line.replace("-", "")                  # join hyphenated compounds
    line = re.sub(r"[,;:.()!?+'\"\u2018\u2019\u201c\u201d\u0303]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


for key, (src, font) in SCRIPTS.items():
    cmap = TTFont(font).getBestCmap()
    with open(src, encoding="utf-8") as f:
        raw = [l for l in f if l.strip()]
    out, seen = [], set()
    for l in raw:
        t = clean(l)
        if not t or not (MIN_LEN <= len(t) <= MAX_LEN) or t in seen:
            continue
        if all(c == " " or c in JOINERS or ord(c) in cmap for c in t):
            seen.add(t)
            out.append(t)
    with open(f"data/texts/{key}_clean.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"{key}: {len(out)} usable of {len(raw)} lines")
    for s in out[:2]:
        print("   ", s[:70])
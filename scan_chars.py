import json
from collections import Counter

RANGES = {
    "devanagari": (0x0900, 0x097F),
    "modi": (0x11600, 0x1165F),
    "sharada": (0x11180, 0x111DF),
}

for s, (lo, hi) in RANGES.items():
    odd = Counter()
    for sp in ("train", "validation", "test"):
        path = f"hf_dataset/{s}/{sp}/metadata.jsonl"
        for line in open(path, encoding="utf-8"):
            for ch in json.loads(line)["text"]:
                c = ord(ch)
                if lo <= c <= hi or ch.isspace() or c < 128 or c in (0x200C, 0x200D):
                    continue
                odd[f"U+{c:X}"] += 1
    print(s, sum(odd.values()), odd.most_common(8))
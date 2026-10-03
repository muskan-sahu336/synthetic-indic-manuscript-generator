import json
import os
import shutil

SRC, DST = "output", "hf_dataset"
SCRIPTS = ["devanagari", "modi", "sharada"]
EXPECTED = {"train": 85, "validation": 10, "test": 5}

CARD = """---
license: cc-by-4.0
task_categories:
- image-to-text
tags:
- synthetic
- ocr
- historical-documents
- devanagari
- modi
- sharada
configs:
- config_name: devanagari
  data_dir: devanagari
- config_name: modi
  data_dir: modi
- config_name: sharada
  data_dir: sharada
---

# Synthetic Indic Manuscript Images (Devanagari, Modi, Sharada)

Synthetically generated manuscript-style page images paired with their ground-truth text, for OCR / handwriting-recognition research on Indic scripts. All images are generated, not scans of real manuscripts.

## Configs and splits

Three configs, one per script. Each has 100 pages split 85 / 10 / 5.

| Config | Script | train | validation | test |
|---|---|---|---|---|
| devanagari | Devanagari | 85 | 10 | 5 |
| modi | Modi | 85 | 10 | 5 |
| sharada | Sharada | 85 | 10 | 5 |

```python
from datasets import load_dataset
ds = load_dataset("muskan336/indic-manuscripts-synthetic", "modi")
```

## Fields

- `image`: the rendered page (PNG).
- `text`: the page transcription in Unicode. Each line of `text` is one row of the main text on the page. Where a row is split by gaps such as string holes, the pieces are joined left to right on one line. If the page has a marginal note, its words are added as the last line.

Each page's transcription is also stored as a `.md` file next to its image.

## How it was generated

Pages are rendered by a Python pipeline from one text file per script (source texts supplied with the assignment). Devanagari is set in Kalam, a handwriting-style font; Modi and Sharada are set in Noto Sans Modi and Noto Sans Sharada. Each page places lines of text on a simulated manuscript background (aged paper or palm leaf) with ink-tone variation, occasional red highlighted words, and effects such as ink bleed and fading, smudges, blots, fold marks, and paper texture and shadow. Words get handwriting-like variation (slant, wavy baselines, small changes in size, rotation and pen weight). Palm-leaf pages may be laid out in several columns separated by string holes. Some pages have ruled margin lines, a marginal note or string holes. Each script has 100 pages, split 85 / 10 / 5 into train / validation / test.

## Limitations

- Synthetic data: page appearance is simulated and will differ from real manuscripts.
- Ruled margin lines and string holes are not part of the `text` annotation. Marginal notes are: paper pages can have a short rotated note (2 to 4 words) in the left margin, and its words are the last line of `text`. The note's words are taken from the source text just after the main text, so they are not a separate annotation. 34 to 41 pages per script have a note.
- The text of each page is a run of consecutive lines from the source, cut where the page runs out of room, so a page can end in the middle of a sentence.
- 98 of the 100 Devanagari pages contain verse markers such as `।६।`; Modi and Sharada pages have none. The markers are inserted by the generator, count up from a random starting number, and do not match the verse numbers of the source. They are drawn on the page and included in `text`.
- Text correctness depends on the source text and the fonts; it has not been reviewed by a paleography expert. The transcription is the source text exactly as drawn, so errors in the source are reproduced.
- The Hugging Face viewer may show some Modi and Sharada characters as boxes or emoji, because browsers often lack fonts for these scripts. The underlying text is valid Unicode (Modi U+11600 to U+1165F, Sharada U+11180 to U+111DF).

## License

Released under CC BY 4.0. Source texts: provided with the assignment.
"""

shutil.rmtree(DST, ignore_errors=True)
for s in SCRIPTS:
    for sp, n_expected in EXPECTED.items():
        src = os.path.join(SRC, s, sp)
        dst = os.path.join(DST, s, sp)
        os.makedirs(dst)
        pngs = sorted(f for f in os.listdir(src) if f.endswith(".png"))
        assert len(pngs) == n_expected, f"{s}/{sp}: {len(pngs)} images, expected {n_expected}"
        rows = []
        for p in pngs:
            md = os.path.join(src, p[:-4] + ".md")
            assert os.path.exists(md), f"missing {md}"
            shutil.copy2(os.path.join(src, p), dst)
            shutil.copy2(md, dst)
            with open(md, encoding="utf-8", newline="") as f:
                rows.append({"file_name": p, "text": f.read()})
        with open(os.path.join(dst, "metadata.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{s}/{sp}: {len(rows)} images + md + metadata.jsonl")

with open(os.path.join(DST, "README.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write(CARD)
print("done ->", DST)
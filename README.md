# Synthetic Indic Manuscript Generator

A Python pipeline that renders synthetic historical manuscript folios from plain text files and writes a matching ground-truth annotation (`.md`) for every page. It supports three scripts, **Devanagari**, **Modi** and **Sharada**, and any other script can be added through the config file.

**Hugging Face dataset:** [`muskan336/indic-manuscripts-synthetic`](https://huggingface.co/datasets/muskan336/indic-manuscripts-synthetic)
(subsets `devanagari`, `modi`, `sharada`; each with `train` / `validation` / `test` = 85 / 10 / 5 pages)

## What it generates

- **Backgrounds:** aged handmade paper and palm leaf, with texture and aging variation.
- **Text:** shaped with HarfBuzz and rendered with handwriting-like variation: slant, wavy baselines, and small changes in size, rotation, pen weight and ink tone.
- **Layout:** several lines per page, multiple columns where string holes interrupt a line, small rotated marginal notes in the left margin, and verse markers such as `।६।` (where the font has the glyphs).
- **Highlights:** occasional red words.
- **Artifacts:** ink bleed, fading, smudges, blots, folds and page warping (see `src/effects.py`).
- **Constraint check:** every page is checked so that no text falls outside the allowed region or touches the page edge. Pages that fail are re-rendered (up to `retry_limit` times).
- **Reproducible:** pages are seeded from `seed`, the script name, the page index and the attempt number.

## Installation

```bash
git clone https://github.com/muskan-sahu336/synthetic-indic-manuscript-generator.git
cd synthetic-indic-manuscript-generator
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

You also need:

- **Fonts** in `fonts/` (names are set in `config.yaml`): `Kalam-Regular.ttf` for Devanagari (with `NotoSansDevanagari-Regular.ttf` as fallback), `NotoSansModi-Regular.ttf` and `NotoSansSharada-Regular.ttf`.
- **Text files** in `data/texts/`: `devanagari_clean.txt`, `modi_clean.txt`, `sharada_clean.txt`, one line of source text per line.

## Usage

```bash
python generate.py
```

This generates 100 pages for every script in `config.yaml` into `output/`. Options:

```bash
python generate.py --script modi            # one script only (devanagari | modi | sharada | any key in config)
python generate.py --count 20               # pages per script
python generate.py --seed 7                 # different random seed
python generate.py --config my_config.yaml  # different config
python generate.py --output out2            # different output folder
```

A summary table (pages per split, retries, failures) is printed at the end.

### Output layout

```
output/
  devanagari/
    train/        devanagari_000.png  devanagari_000.md  ...
    validation/   ...
    test/         ...
    manifest.jsonl
  modi/ ...
  sharada/ ...
```

Each `.md` file is plain text:

- one line per row of the main text on the page, in reading order;
- if a row is split by gaps such as string holes, the pieces are joined left to right on one line;
- if the page has a marginal note, its words are the last line (`export.include_notes_in_md`).

`manifest.jsonl` records, for each page, its split, background kind (`paper` or `palm`), number of lines, number of notes and the corpus lines used.

### Publishing to Hugging Face

```bash
python prepare_hf.py     # builds hf_dataset/ (images, .md, metadata.jsonl, dataset card)
python upload_hf.py      # uploads hf_dataset/ (run `huggingface-cli login` first)
```

## Configuration (`config.yaml`)

| Key | Meaning |
|---|---|
| `seed`, `count` | random seed; pages per script |
| `splits` | train / validation / test fractions (0.85 / 0.10 / 0.05) |
| `retry_limit` | re-render attempts for a page that fails the overflow check |
| `min_block_lines` | minimum corpus lines per page (aborts if the text file is too short) |
| `backgrounds` | page sizes for paper and palm leaf, side of the margin rules |
| `layout` | line ranges per page for paper and palm leaf; `margin_notes` on / off |
| `export.include_notes_in_md` | whether marginal-note words are written to the `.md` |
| `overflow` | threshold and border width for the overflow check |
| `effects` | ink bleed, fading, dropout, smudges, blots, folds (overrides defaults in `src/effects.py`) |
| `style` | handwriting variation (overrides defaults in `src/renderer.py`) |
| `scripts` | one entry per script: `font`, `font_fallback`, `corpus`, `digit_base`, `paper_prob` |

### Adding a new script

Add an entry under `scripts:` in `config.yaml` with a font file, a text file, the code point of the script's digit zero (`digit_base`) and `paper_prob` (probability of a paper page; otherwise palm leaf). Then run `python generate.py --script <name>`.

## Repository structure

```
generate.py          entry point: python generate.py
config.yaml          all configurable parameters
src/
  backgrounds.py     paper and palm-leaf backgrounds, margin rules
  renderer.py        HarfBuzz shaping, handwriting-style word rendering, ink colours
  layout.py          line wrapping, columns, marginal notes, verse markers, red words
  effects.py         ink and paper effects, page finishing
  export.py          overflow check, .png / .md writing
prepare_hf.py        builds the Hugging Face dataset folder and dataset card
upload_hf.py         uploads it to Hugging Face
prepare_texts.py     prepares the text files in data/texts/
scan_chars.py, check_*.py, test_*.py   development checks and tests
fonts/               font files
data/texts/          source text files (one per script)
```

## Dataset

The dataset on Hugging Face has three subsets, one per script, and each has `train` (85), `validation` (10) and `test` (5) splits. Each row has an `image` (PNG) and the page `text`. The dataset card lists the known limitations: the data is synthetic, marginal-note words are included in `text`, Devanagari pages contain generated verse markers, and the transcription is the source text exactly as drawn.

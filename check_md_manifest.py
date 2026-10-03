"""Checks every page's .md against output/<script>/manifest.jsonl.
Run from the project folder:  python check_md_manifest.py   (or --output other_dir)

For each page: number of non-blank lines in the .md must equal n_lines + n_notes
(main lines + the marginal note, which export.build_md puts last).
Also reports: pages per background kind, pages with a note, notes on palm pages
(should be none), and pages containing generator verse markers like ।६।."""
import argparse, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--output", default="output")
args = ap.parse_args()

digits = "".join(chr(b + i) for b in (0x0966, 0x11650, 0x111D0) for i in range(10))
marker = re.compile("\u0964[" + digits + "]+\u0964")

for script in ("devanagari", "modi", "sharada"):
    mpath = os.path.join(args.output, script, "manifest.jsonl")
    if not os.path.exists(mpath):
        print(f"{script}: no manifest at {mpath}")
        continue
    rows = [json.loads(l) for l in open(mpath, encoding="utf-8") if l.strip()]
    kinds, with_note, palm_notes, bad, with_marker = {}, 0, 0, [], 0
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        md = os.path.join(args.output, script, r["file"][:-4] + ".md")
        text = open(md, encoding="utf-8", newline="").read()
        n = len([l for l in text.split("\n") if l.strip()])
        if r["n_notes"]:
            with_note += 1
            if r["kind"] != "paper":
                palm_notes += 1
        if n != r["n_lines"] + r["n_notes"]:
            bad.append((r["file"], n, r["n_lines"], r["n_notes"]))
        if marker.search(text):
            with_marker += 1
    print(f"{script}: {len(rows)} pages, kinds {kinds}, with marginal note {with_note}, "
        f"notes on non-paper pages {palm_notes}, pages with ।N। markers {with_marker}, "
        f"line-count mismatches {len(bad)}")
    for b in bad[:10]:
        print("   mismatch:", b)
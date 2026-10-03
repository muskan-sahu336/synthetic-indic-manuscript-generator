import argparse
import json
import os
import shutil
import time
import zlib

import numpy as np
import yaml

from src.backgrounds import make_paper, make_palm_leaf, add_margin_rules
from src.layout import layout_page, make_marker, register_digit_base
from src.effects import finish_page, EFFECTS
from src.renderer import STYLE
from src.export import check_overflow, save_sample

SPLITS = ("train", "validation", "test")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_font(name, spec):
    primary, fallback = spec.get("font"), spec.get("font_fallback")
    if primary and os.path.exists(primary):
        return primary
    if primary:
        print(f"WARNING [{name}]: font not found: {primary}")
    if fallback and os.path.exists(fallback):
        print(f"WARNING [{name}]: using fallback font {fallback}")
        return fallback
    raise SystemExit(f"[{name}] no usable font (tried {primary}, {fallback})")


def load_corpus(path):
    if not os.path.exists(path):
        raise SystemExit(f"corpus not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def split_counts(count, fr):
    n_val = round(count * fr["validation"])
    n_test = round(count * fr["test"])
    if count >= 3:
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    return {"train": count - n_val - n_test, "validation": n_val, "test": n_test}


def split_plan(count, cfg, seed, script):
    """page index -> split, via a seeded shuffle."""
    sizes = split_counts(count, cfg["splits"])
    ss = np.random.SeedSequence([seed, zlib.crc32(script.encode()), 999_999])
    perm = np.random.default_rng(ss).permutation(count)
    plan, pos = {}, 0
    for sp in SPLITS:
        for i in perm[pos:pos + sizes[sp]]:
            plan[int(i)] = sp
        pos += sizes[sp]
    return plan


def page_rng(seed, script, idx, attempt):
    return np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(script.encode()), idx, attempt]))


def render_page(script, spec, font, block, idx, attempt, cfg):
    rng = page_rng(cfg["seed"], script, idx, attempt)
    bg, lay = cfg["backgrounds"], cfg["layout"]
    effects = {**EFFECTS, **(cfg.get("effects") or {})}
    style = {**STYLE, **(cfg.get("style") or {})}

    kind = "paper" if rng.random() < spec["paper_prob"] else "palm"
    if kind == "paper":
        w, h = bg["paper_size"]
        res = add_margin_rules(make_paper(w, h, rng), rng, bg.get("rules_side", "right"))
        lo, hi = lay["paper_lines"]
    else:
        w, h = bg["palm_size"]
        res = make_palm_leaf(w, h, rng)
        lo, hi = lay["palm_lines"]
    n_lines = int(rng.integers(lo, hi + 1))

    out = layout_page(res, font, block, 0, rng, script, n_lines=n_lines,
                      notes=lay["margin_notes"], style=style)
    fin = finish_page(res, out["layer"], rng, extra_boxes=out["regions"]["notes"], cfg=effects)
    ok, n_out, n_edge = check_overflow(fin, cfg["overflow"]["thresh"], cfg["overflow"]["border"])
    if not out["lines"]:
        ok = False
    return {"ok": ok, "kind": kind, "fin": fin, "out": out,
            "wrapped": out["next_idx"] > len(block)}


def run_script(script, cfg, count):
    spec = cfg["scripts"][script]
    font = resolve_font(script, spec)
    register_digit_base(script, spec["digit_base"])
    if make_marker(script, 12, font) is None:
        print(f"WARNING [{script}]: font lacks verse-marker glyphs; markers will be skipped")
    corpus = load_corpus(spec["corpus"])
    bs = len(corpus) // count
    if bs < cfg["min_block_lines"]:
        raise SystemExit(f"[{script}] corpus has {len(corpus)} lines; {count} pages leave only "
                         f"{bs} lines per page (min {cfg['min_block_lines']}).")

    root = os.path.join(cfg["output"], script)
    for sp in SPLITS:
        shutil.rmtree(os.path.join(root, sp), ignore_errors=True)
    plan = split_plan(count, cfg, cfg["seed"], script)

    stats = {"train": 0, "validation": 0, "test": 0, "retries": 0, "failed": [], "wrapped": 0}
    manifest, t_script = [], time.time()
    for n, idx in enumerate(range(count), 1):
        t0 = time.time()
        block = corpus[idx * bs:(idx + 1) * bs]
        sp = plan[idx]
        result, attempt = None, 0
        while attempt <= cfg["retry_limit"]:
            r = render_page(script, spec, font, block, idx, attempt, cfg)
            if r["ok"]:
                result = r
                break
            attempt += 1
        if result is None:
            stats["failed"].append(idx)
            print(f"[{script}] {n}/{count} idx={idx:03d} FAILED after {attempt} attempts")
            continue

        stem = f"{script}_{idx:03d}"
        out = result["out"]
        save_sample(os.path.join(root, sp), stem, result["fin"]["image"], out["lines"],
                    out["notes"], include_notes=cfg["export"]["include_notes_in_md"])
        stats[sp] += 1
        stats["retries"] += attempt
        stats["wrapped"] += int(result["wrapped"])
        manifest.append({"index": idx, "split": sp, "file": f"{sp}/{stem}.png",
                         "kind": result["kind"], "attempt": attempt,
                         "corpus_lines": [idx * bs, (idx + 1) * bs],
                         "n_lines": len(out["lines"]), "n_notes": len(out["notes"])})
        print(f"[{script}] {n}/{count} idx={idx:03d} {sp:<10} {result['kind']:<5} "
              f"lines={len(out['lines'])} retries={attempt} "
              f"{'WRAPPED ' if result['wrapped'] else ''}{time.time() - t0:.1f}s", flush=True)

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    stats["seconds"] = time.time() - t_script
    return stats


def main():
    ap = argparse.ArgumentParser(description="Synthetic Indic manuscript generator")
    ap.add_argument("--script", default="all", help="devanagari|modi|sharada|all (or any key in config)")
    ap.add_argument("--count", type=int, default=None, help="pages per script (default from config)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output:
        cfg["output"] = args.output
    count = args.count or cfg["count"]
    names = list(cfg["scripts"]) if args.script == "all" else [args.script]
    for s in names:
        if s not in cfg["scripts"]:
            raise SystemExit(f"unknown script '{s}'; available: {', '.join(cfg['scripts'])}")

    all_stats = {s: run_script(s, cfg, count) for s in names}

    print("\n=== SUMMARY ===")
    print(f"{'script':<12}{'train':>7}{'valid':>7}{'test':>6}{'retries':>9}{'failed':>8}{'wrapped':>9}{'time(s)':>9}")
    for s, st in all_stats.items():
        print(f"{s:<12}{st['train']:>7}{st['validation']:>7}{st['test']:>6}{st['retries']:>9}"
              f"{len(st['failed']):>8}{st['wrapped']:>9}{st['seconds']:>9.0f}")
        if st["failed"]:
            print(f"  failed page indices: {st['failed']}")


if __name__ == "__main__":
    main()
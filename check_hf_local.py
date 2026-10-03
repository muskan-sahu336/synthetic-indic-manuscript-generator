from datasets import load_dataset

for name in ("devanagari", "modi", "sharada"):
    ds = load_dataset("hf_dataset", name)
    print(name, {k: len(v) for k, v in ds.items()}, ds["train"].column_names)
    print("   ", repr(ds["train"][0]["text"][:60]))
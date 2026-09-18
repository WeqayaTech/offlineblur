#!/usr/bin/env python3
"""Export public Hugging Face face datasets into files + manifest CSVs for train_demographics.py
(no Kaggle account needed). Schema of the CSV: path,age,age_lo,age_hi,gender.

    python3 hf_export.py --root /root/offlineblur_v2/data --manifests /root/offlineblur_v2/manifests \
        [--utkface 0] [--fairface 0] [--celeba 40000]        # 0 = all rows; CelebA is gender-only, capped by default

  utkface   nu-delta/utkface                 23 k, exact age 0..116, gender          → utk.csv
  fairface  HuggingFaceM4/FairFace (1.25)    97 k, age groups, gender, diverse       → fairface.csv  (stands in for Adience)
  celeba    tpremoli/CelebA-attrs            162 k, gender only (Male attribute)     → celeba.csv
"""
import argparse
import csv
from pathlib import Path

FAIRFACE_GROUPS = {"0-2": (0, 2), "3-9": (3, 9), "10-19": (10, 19), "20-29": (20, 29), "30-39": (30, 39), "40-49": (40, 49),
                   "50-59": (50, 59), "60-69": (60, 69), "more than 70": (70, 90)}


def save(img, path):
    if path.exists():
        return
    img.convert("RGB").save(path, quality=95)


def export(name, split, out_dir, rows_fn, cap, config=None):
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    n = len(ds) if cap <= 0 else min(cap, len(ds))
    out = []
    for i in range(n):
        r = ds[i]
        p = out_dir / f"{split}_{i:07d}.jpg"
        row = rows_fn(r)
        if row is None:
            continue
        save(r["image"], p)
        out.append([str(p), *row])
        if i % 5000 == 0:
            print(f"[{name}] {i}/{n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--utkface", type=int, default=0)
    ap.add_argument("--fairface", type=int, default=0)
    ap.add_argument("--celeba", type=int, default=40000)
    ap.add_argument("--skip", default="", help="comma list of datasets to skip")
    a = ap.parse_args()
    root, man = Path(a.root), Path(a.manifests)
    man.mkdir(parents=True, exist_ok=True)
    skip = set(a.skip.split(",")) if a.skip else set()

    def write(name, rows):
        with open(man / f"{name}.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"[{name}] {len(rows)} rows → {man / (name + '.csv')}")

    if "utkface" not in skip:
        def utk(r):
            g = {"Male": 0, "Female": 1}.get(r["gender"], -1)
            age = int(r["age"])
            if g < 0 or not 0 <= age <= 100:
                return None
            return [age, age, age, g]
        write("utk", export("nu-delta/utkface", "train", root / "utkface", utk, a.utkface))
    if "fairface" not in skip:
        from datasets import load_dataset_builder
        rows = []
        for split in ("train", "validation"):
            ds_names = None

            def ff(r, _n=[None]):
                if _n[0] is None:
                    from datasets import load_dataset
                    d = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")
                    _n[0] = (d.features["age"].names, d.features["gender"].names)
                ages, genders = _n[0]
                grp = FAIRFACE_GROUPS.get(ages[r["age"]])
                g = {"Male": 0, "Female": 1}.get(genders[r["gender"]], -1)
                if grp is None or g < 0:
                    return None
                return [-1, grp[0], grp[1], g]
            rows += export("HuggingFaceM4/FairFace", split, root / "fairface", ff, a.fairface, config="1.25")
        write("fairface", rows)
    if "celeba" not in skip:
        def cel(r):
            g = 0 if int(r["Male"]) == 1 else 1
            return [-1, -1, -1, g]
        write("celeba", export("tpremoli/CelebA-attrs", "train", root / "celeba", cel, a.celeba))
    allp = man / "all.csv"
    with open(allp, "w", newline="") as out:
        for f in sorted(man.glob("*.csv")):
            if f.name != "all.csv":
                out.write(f.read_text())
    print(f"[all] → {allp}: {sum(1 for _ in open(allp))} rows")


if __name__ == "__main__":
    main()

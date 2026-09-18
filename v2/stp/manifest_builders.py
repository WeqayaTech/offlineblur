#!/usr/bin/env python3
"""Manifest builders for the demographic head. One CSV, any mix of datasets:

    path,age,age_lo,age_hi,gender        age -1 = unknown; age_lo/age_hi = group range (Adience); gender 0 = male, 1 = female, -1 = unknown

    python3 manifest_builders.py utkface /workspace/data/UTKFace          > manifests/utk.csv      (file name: [age]_[gender]_[race]_[date].jpg; gender 0=male 1=female)
    python3 manifest_builders.py adience /workspace/data/adience           > manifests/adience.csv  (fold_*_data.txt + faces/ or aligned/)
    python3 manifest_builders.py celeba  /workspace/data/celeba            > manifests/celeba.csv   (list_attr_celeba.txt + img_align_celeba/; gender only)
    cat manifests/*.csv > manifests/all.csv
Datasets are downloaded by you (Kaggle / official pages); nothing here fetches them.
"""
import csv
import sys
from pathlib import Path

ADIENCE_GROUPS = {"(0, 2)": (0, 2), "(4, 6)": (4, 6), "(8, 12)": (8, 12), "(8, 23)": (8, 23), "(15, 20)": (15, 20),
                  "(25, 32)": (25, 32), "(27, 32)": (27, 32), "(38, 43)": (38, 43), "(38, 48)": (38, 48), "(48, 53)": (48, 53),
                  "(60, 100)": (60, 100)}


def utkface(root):
    for p in sorted(Path(root).rglob("*.jpg")):
        parts = p.name.split("_")
        try:
            age, gender = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue
        if not (0 <= age <= 100 and gender in (0, 1)):
            continue
        yield [str(p), age, age, age, gender]


def adience(root):
    root = Path(root)
    img_roots = [d for d in (root / "aligned", root / "faces") if d.exists()] or [root]
    for fold in sorted(root.glob("fold_*_data.txt")):
        with open(fold) as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            for r in rd:
                g = {"m": 0, "f": 1}.get(r.get("gender", ""), -1)
                grp = ADIENCE_GROUPS.get(r.get("age", ""))
                if grp is None:
                    try:
                        a = int(r["age"]); grp = (a, a)
                    except ValueError:
                        grp = None
                cands = [ir / r["user_id"] / f"landmark_aligned_face.{r['face_id']}.{r['original_image']}" for ir in img_roots]
                cands += [ir / r["user_id"] / f"coarse_tilt_aligned_face.{r['face_id']}.{r['original_image']}" for ir in img_roots]
                p = next((c for c in cands if c.exists()), None)
                if p is None or (g < 0 and grp is None):
                    continue
                if grp is None:
                    yield [str(p), -1, -1, -1, g]
                else:
                    yield [str(p), -1, grp[0], grp[1], g]


def celeba(root):
    root = Path(root)
    attr = root / "list_attr_celeba.txt"
    img = root / "img_align_celeba"
    with open(attr) as fh:
        lines = fh.read().splitlines()
    header = lines[1].split()
    mi = header.index("Male")
    for line in lines[2:]:
        p = line.split()
        if len(p) <= mi + 1:
            continue
        g = 0 if p[mi + 1] == "1" else 1
        path = img / p[0]
        if path.exists():
            yield [str(path), -1, -1, -1, g]


if __name__ == "__main__":
    kind, root = sys.argv[1], sys.argv[2]
    gen = {"utkface": utkface, "adience": adience, "celeba": celeba}[kind](root)
    w = csv.writer(sys.stdout)
    n = 0
    for row in gen:
        w.writerow(row)
        n += 1
    print(f"{kind}: {n} rows", file=sys.stderr)

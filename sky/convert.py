import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SPLITS = (("train", "train"), ("valid", "val"), ("test", "test"))


def names_of(names):
    return list(names)


def load_meta(yaml_path):
    raw = yaml.safe_load(Path(yaml_path).read_text())
    root = Path(raw["path"])
    if not root.is_absolute():
        root = (Path(yaml_path).resolve().parent / root).resolve()
    raw = dict(raw)
    raw["_root"] = root
    return raw


def split_dir(meta, split):
    return Path(meta["_root"]) / meta[split]


def label_dir_for(images):
    parts = list(Path(images).parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts)


def iter_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return
    yield from sorted(folder.glob("*.jpg"))


def link(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return False
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError:
        shutil.copy2(src, dst)
    return True


def read_label(path, width, height):
    boxes = []
    path = Path(path)
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = line.split()[:5]
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        boxes.append(
            (
                int(cls),
                [
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ],
            )
        )
    return boxes


def yolo_line(ann, width, height):
    x, y, bw, bh = ann["bbox"]
    x1 = min(max(x, 0.0), width)
    y1 = min(max(y, 0.0), height)
    x2 = min(max(x + bw, 0.0), width)
    y2 = min(max(y + bh, 0.0), height)
    bw, bh = x2 - x1, y2 - y1
    if bw < 1 or bh < 1:
        return None
    cls = ann["category_id"] - 1
    return f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {bw / width:.6f} {bh / height:.6f}"


def convert_split(src, folder, dst_name, dst):
    data = json.loads((src / f"{folder}.json").read_text())
    ids = [c["id"] for c in data["categories"]]
    if ids != [1, 2, 3]:
        raise SystemExit(f"{folder}.json категории {ids}, ждал 1,2,3")
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)
    out_img = Path(dst) / "images" / dst_name
    out_lbl = Path(dst) / "labels" / dst_name
    if out_img.exists():
        shutil.rmtree(out_img)
    if out_lbl.exists():
        shutil.rmtree(out_lbl)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)
    counts = Counter()
    skipped = 0
    for image in data["images"]:
        src_img = src / folder / image["file_name"]
        link(src_img, out_img / image["file_name"])
        lines = []
        for ann in by_image[image["id"]]:
            row = yolo_line(ann, image["width"], image["height"])
            if row is None:
                skipped += 1
                continue
            lines.append(row)
            counts[int(row.split()[0])] += 1
        text = ("\n".join(lines) + "\n") if lines else ""
        (out_lbl / (Path(image["file_name"]).stem + ".txt")).write_text(text)
    print(f"{folder}: {len(data['images'])} картинок, {sum(counts.values())} боксов, обрезал {skipped}")
    return counts


def convert(src, dst, yaml_path):
    if src:
        src = Path(src)
    else:
        from sky.download import ensure_raw

        src = ensure_raw()
    if not (src / "train.json").is_file():
        raise SystemExit(f"в {src} нет train.json")
    names = [line.strip() for line in (src / "classes.txt").read_text().splitlines() if line.strip()]
    print("src", src)
    print("классы", names)
    dst = Path(dst)
    totals = Counter()
    for folder, dst_name in SPLITS:
        totals.update(convert_split(src, folder, dst_name, dst))
    for i, name in enumerate(names):
        print(f"  {i} {name}: {totals[i]}")

    train_names = {p.name for p in (dst / "images" / "train").glob("*.jpg")}
    test_names = {p.name for p in (dst / "images" / "test").glob("*.jpg")}
    both = train_names & test_names
    if both:
        raise SystemExit(f"test попал в train, например {next(iter(both))}")

    meta = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print("yaml:", path)
    return path

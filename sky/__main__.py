"""
python -m sky convert
python -m sky convert --src data/raw --dst data/yolo
python -m sky train --config configs/yolo26n.yaml
python -m sky train --config configs/yolo26n.yaml --no-wandb --set epochs=1 batch=2
python -m sky yolov5 --config configs/yolov5n.yaml
"""
import argparse
from pathlib import Path

import yaml


def parse_set(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def load_yaml(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(overrides or {})
    return cfg


def main():
    parser = argparse.ArgumentParser(prog="sky", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert")
    p.add_argument("--src")
    p.add_argument("--dst", default="data/yolo")
    p.add_argument("--yaml", default="data/skyfusion.yaml")

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*")
    p.add_argument("--no-wandb", action="store_true")

    p = sub.add_parser("tiles")
    p.add_argument("weights")
    p.add_argument("--data", default="data/skyfusion.yaml")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.5)

    p = sub.add_parser("yolov5")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*")
    p.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    if args.command == "convert":
        from sky.convert import convert

        convert(args.src, args.dst, args.yaml)
    elif args.command == "train":
        from sky.train import train

        overrides = parse_set(args.set)
        if args.no_wandb:
            overrides["wandb"] = False
        cfg = load_yaml(args.config, overrides)
        if not cfg.get("name"):
            cfg["name"] = Path(args.config).stem
        train(cfg)
    elif args.command == "tiles":
        from sky.tiles import tiles

        tiles(args.weights, args.data, args.tile, args.iou)
    elif args.command == "yolov5":
        from sky.yolov5 import yolov5

        overrides = parse_set(args.set)
        if args.no_wandb:
            overrides["wandb"] = False
        cfg = load_yaml(args.config, overrides)
        if not cfg.get("name"):
            cfg["name"] = Path(args.config).stem
        yolov5(cfg)


if __name__ == "__main__":
    main()

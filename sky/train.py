import os
from pathlib import Path

import yaml

from sky.convert import iter_images, label_dir_for, link, load_meta, split_dir

SKIP = {
    "wandb",
    "wandb_project",
    "entity",
    "eval_split",
    "merge_val",
    "init_from",
    "load",
    "model",
}


def _place(src, folder):
    src = Path(src)
    dest = folder / src.name
    if dest.exists() or dest.is_symlink():
        try:
            same = dest.resolve() == src.resolve()
        except OSError:
            same = False
        if not same:
            dest = folder / f"{src.stem}_b{src.suffix}"
    if not dest.exists() and not dest.is_symlink():
        link(src, dest)
    return dest


def merge_val(yaml_path):
    meta = load_meta(yaml_path)
    root = Path(meta["_root"])
    full_img = root / "images" / "train_full"
    full_lbl = root / "labels" / "train_full"
    full_img.mkdir(parents=True, exist_ok=True)
    full_lbl.mkdir(parents=True, exist_ok=True)
    test_resolved = set()
    if meta.get("test"):
        test_resolved = {p.resolve() for p in iter_images(split_dir(meta, "test"))}
    for split in ("train", "val"):
        images = split_dir(meta, split)
        labels = label_dir_for(images)
        for img in iter_images(images):
            if img.resolve() in test_resolved:
                raise SystemExit(f"test в train_full: {img.name}")
            dest = _place(img, full_img)
            lab = labels / (img.stem + ".txt")
            lab_dst = full_lbl / (dest.stem + ".txt")
            if lab.exists():
                if lab_dst.exists() or lab_dst.is_symlink():
                    lab_dst.unlink()
                link(lab, lab_dst)
            elif not lab_dst.exists():
                lab_dst.write_text("")
    n = sum(1 for _ in iter_images(full_img))
    out = {
        "path": str(root),
        "train": "images/train_full",
        "val": meta["val"],
        "names": meta["names"],
    }
    if meta.get("test"):
        out["test"] = meta["test"]
    dest = Path(yaml_path).with_name("skyfusion_full.yaml")
    dest.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"train_full: {n} картинок, {dest}")
    return dest


def _setup_wandb(cfg):
    enabled = cfg.get("wandb", True) and bool(os.environ.get("WANDB_API_KEY"))
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb выключен")
        return
    os.environ.pop("WANDB_MODE", None)
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_ENTITY"] = cfg.get("entity") or "georgy-shulyndin-cu"
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS["wandb"] = True
    except Exception as e:
        print("wandb settings:", e)
    _point_wandb(cfg.get("wandb_project") or "skyfusion")


def _point_wandb(project_name):
    try:
        import ultralytics.utils.callbacks as cb
        import ultralytics.utils.callbacks.wb as wb_cb
    except Exception as e:
        print("wandb:", e)
        return
    orig = getattr(wb_cb, "on_pretrain_routine_end", None)
    if orig is None or getattr(wb_cb, "wb", None) is None:
        print("wandb callback нет")
        return

    def wrapped(trainer):
        folder = trainer.args.project
        trainer.args.project = project_name
        try:
            return orig(trainer)
        finally:
            trainer.args.project = folder

    hooks = cb.default_callbacks.get("on_pretrain_routine_end", [])
    replaced = False
    for i, fn in enumerate(list(hooks)):
        if fn is orig or getattr(fn, "__module__", "").endswith(".wb"):
            hooks[i] = wrapped
            replaced = True
    if not replaced:
        hooks.append(wrapped)
    print("wandb project", project_name)


def _wandb_target(cfg):
    entity = cfg.get("entity") or os.environ.get("WANDB_ENTITY") or "georgy-shulyndin-cu"
    project = cfg.get("wandb_project") or "skyfusion"
    return entity, project


def _artifact_name(cfg):
    path = Path(cfg.get("init_from") or "")
    if path.parent.name == "weights":
        return path.parent.parent.name
    return cfg.get("name")


def _log_weights(cfg, best):
    if not cfg.get("wandb", True) or not os.environ.get("WANDB_API_KEY"):
        return
    if os.environ.get("WANDB_MODE") == "disabled":
        return
    try:
        import wandb

        entity, project = _wandb_target(cfg)
        run = wandb.init(entity=entity, project=project, name=cfg["name"], job_type="weights", reinit=True)
        art = wandb.Artifact(cfg["name"], type="model")
        art.add_file(str(best))
        run.log_artifact(art)
        run.finish()
        print("артефакт", cfg["name"])
    except Exception as e:
        print("веса не залиты:", e)


def _pull_weights(cfg):
    import wandb

    entity, project = _wandb_target(cfg)
    name = _artifact_name(cfg)
    art = wandb.Api().artifact(f"{entity}/{project}/{name}:latest")
    folder = Path(art.download(root=str(Path("artifacts") / name)))
    found = list(folder.rglob("*.pt"))
    if not found:
        raise SystemExit(f"в артефакте {name} нет pt")
    print("скачаны веса", found[0])
    return found[0]


def _build(cfg):
    from ultralytics import YOLO

    kind = f"{cfg.get('init_from') or ''} {cfg.get('model') or ''}".lower()
    cls = YOLO
    if "rtdetr" in kind:
        from ultralytics import RTDETR

        cls = RTDETR
    if cfg.get("init_from"):
        path = Path(cfg["init_from"])
        if not path.exists():
            print("локально нет", path)
            path = _pull_weights(cfg)
        return cls(str(path))
    model = cls(cfg["model"])
    if cfg.get("load"):
        loaded = model.load(cfg["load"])
        if loaded is not None:
            model = loaded
    return model


def _write_metrics(metrics, dest):
    lines = []
    rd = getattr(metrics, "results_dict", None) or {}
    map50 = rd.get("metrics/mAP50(B)")
    if map50 is None and getattr(metrics, "box", None) is not None:
        map50 = float(metrics.box.map50)
    lines.append(f"mAP50: {float(map50):.4f}" if map50 is not None else "mAP50: ?")
    for key, value in rd.items():
        try:
            lines.append(f"{key}: {float(value):.4f}")
        except (TypeError, ValueError):
            pass
    try:
        box = metrics.box
        idxs = list(metrics.ap_class_index)
        for i, ap in zip(idxs, box.ap50):
            name = metrics.names[i]
            lines.append(f"mAP50 {name}: {float(ap):.4f}")
    except Exception as e:
        print("per class:", e)
    text = "\n".join(lines) + "\n"
    Path(dest).write_text(text)
    print(text)


def train(cfg):
    _setup_wandb(cfg)
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        raise SystemExit(f"нет {data_path}")
    if cfg.get("merge_val"):
        data_path = merge_val(data_path)

    model = _build(cfg)
    kwargs = {k: v for k, v in cfg.items() if k not in SKIP and v is not None}
    kwargs["data"] = str(data_path)
    kwargs.setdefault("exist_ok", True)
    print("train", cfg.get("init_from") or cfg.get("model"))
    model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if best.exists():
        _log_weights(cfg, best)
    split = cfg.get("eval_split", "test")
    if not split:
        print("eval выключен,", save_dir)
        return save_dir
    print("тест", best)
    eval_model = type(model)(str(best))
    val_kwargs = dict(
        data=str(data_path),
        split=split,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name + "_test",
        exist_ok=True,
    )
    if cfg.get("imgsz"):
        val_kwargs["imgsz"] = cfg["imgsz"]
    metrics = eval_model.val(**val_kwargs)
    _write_metrics(metrics, save_dir / "test_metrics.txt")
    return save_dir

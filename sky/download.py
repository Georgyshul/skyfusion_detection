import zipfile
from pathlib import Path

import requests

YANDEX_URL = "https://disk.yandex.ru/d/kjfAK54D-zPCTA"


def _with_train_json(path):
    path = Path(path)
    if (path / "train.json").is_file():
        return path
    return None


def find_kaggle():
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for folder in root.iterdir():
        found = _with_train_json(folder)
        if found:
            return found
    return None


def download_yandex(dest, url=YANDEX_URL):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "skyfusion.zip"
    print("качаю с яндекса")
    meta = requests.get(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download",
        params={"public_key": url},
        timeout=60,
    )
    meta.raise_for_status()
    href = meta.json().get("href")
    if not href:
        raise SystemExit(f"яндекс не отдал ссылку: {meta.text[:300]}")
    with requests.get(href, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        step = 50 << 20
        with open(archive, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                prev = got
                got += len(chunk)
                if total and prev // step != got // step:
                    print(f"{got / 1e6:.0f} / {total / 1e6:.0f} MB")
    print("распаковка")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()


def ensure_raw():
    local = _with_train_json("data/raw")
    if local:
        return local
    mounted = find_kaggle()
    if mounted:
        print("датасет уже смонтирован:", mounted)
        return mounted
    dest = Path("/kaggle/working/raw") if Path("/kaggle/working").is_dir() else Path("data/raw")
    download_yandex(dest)
    found = _with_train_json(dest)
    if found is None:
        raise SystemExit(f"после распаковки в {dest} нет train.json")
    return found

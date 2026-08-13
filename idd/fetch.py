"""Download the open data used by this project (no account needed, both CC BY 4.0).

  python -m idd.fetch --shards 10 150 300      # shard indices out of 646

  eval:  huggingface.co/datasets/thirdeyelabs/indian-road-benchmark  (dev split, human labels)
  train: huggingface.co/datasets/addu1209/indian-road-dataset         (pseudo-labelled WebDataset shards)
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = "https://huggingface.co/datasets/thirdeyelabs/indian-road-benchmark/resolve/main"
TRAIN = "https://huggingface.co/datasets/addu1209/indian-road-dataset/resolve/main"


def get(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def fetch_benchmark(root: Path) -> None:
    get(f"{BENCH}/annotations/dev.json", root / "bench" / "dev.json")
    get(f"{BENCH}/evaluate.py", root / "bench" / "evaluate.py")
    names = [im["file_name"] for im in json.loads((root / "bench" / "dev.json").read_text())["images"]]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda n: get(f"{BENCH}/images/dev/{n}", root / "bench" / "images" / "dev" / n), names))


def fetch_shards(root: Path, indices: list[int]) -> list[Path]:
    out = []
    for i in indices:
        dst = root / "shards" / f"train-{i:05d}.tar"
        get(f"{TRAIN}/data/train-{i:05d}-of-00646.tar", dst)
        out.append(dst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    ap.add_argument("--shards", nargs="*", type=int, default=[10, 150, 300])
    a = ap.parse_args()
    fetch_benchmark(Path(a.root))
    print([str(p) for p in fetch_shards(Path(a.root), a.shards)])


if __name__ == "__main__":
    main()

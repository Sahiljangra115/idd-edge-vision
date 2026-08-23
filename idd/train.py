"""Fine-tune a small YOLO detector on the prepared (pseudo-labelled) Indian road frames.

  python -m idd.train --data data/yolo/data.yaml --epochs 40
  python -m idd.train --epochs 40 --chunk 20      # stop after 20 epochs; rerun the same line to resume

--chunk N lets a shared GPU interleave: each call trains at most N epochs, keeps an
unstripped runs/<name>/weights/last.pt, and exits with code 3. The next call with the same
--name resumes from it (optimizer, EMA, LR schedule intact). Exit code 0 means training is
complete and weights/<name>_best.pt exists.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

CHUNK_EXIT = 3


def resumable(last: Path) -> bool:
    """True when last.pt still carries optimizer state, i.e. training did not finish."""
    if not last.exists():
        return False
    import torch
    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    return ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="weights/yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name", default="yolo11n_indian")
    ap.add_argument("--fraction", type=float, default=1.0, help="smoke test: train on this share of images")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop patience in epochs; 0 = off (the 259-frame pseudo-label val mAP is noisy, "
                         "and patience 10 stopped run 1 at epoch 24 of 40, before LR decay and the no-mosaic epochs)")
    ap.add_argument("--chunk", type=int, default=0, help="max epochs in this call (0 = no limit)")
    a = ap.parse_args()

    from ultralytics import YOLO
    run_dir = Path("runs") / a.name
    last = run_dir / "weights" / "last.pt"
    if resumable(last):
        print(f"resuming {last}", flush=True)
        model = YOLO(str(last))
        kw = dict(resume=True, workers=a.workers, batch=a.batch)
    else:
        model = YOLO(a.model)
        kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
                  project=str(Path("runs").resolve()), name=a.name, exist_ok=True, device=0,
                  patience=a.patience, seed=0, plots=False, close_mosaic=5, fraction=a.fraction)

    if a.chunk:
        done = {"n": 0}

        def stop_after_chunk(trainer):
            done["n"] += 1
            if done["n"] >= a.chunk and not trainer.stop:
                # last.pt for this epoch is already on disk with optimizer state; leave it unstripped.
                print(f"chunk of {a.chunk} epochs done at epoch {trainer.epoch + 1}/{trainer.epochs}", flush=True)
                sys.stdout.flush()
                os._exit(CHUNK_EXIT)

        model.add_callback("on_fit_epoch_end", stop_after_chunk)

    model.train(**kw)
    best = run_dir / "weights" / "best.pt"
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, Path("weights") / f"{a.name}_best.pt")
    print("saved", Path("weights") / f"{a.name}_best.pt")


if __name__ == "__main__":
    main()

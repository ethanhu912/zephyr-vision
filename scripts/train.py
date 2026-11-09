r"""Train YOLO11 on converted LaRS. Outputs to runs/detect/<name>/.

    python scripts/train.py --check                  # smoke test: 2 epochs, 10% of train (~8 min CPU)
    python scripts/train.py                          # full run, 100 epochs (GPU)
    python scripts/train.py --model yolo11s.pt --name yolo11s_lars
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="smoke test: 2 epochs on 10%% of train (overridable by --epochs/--fraction)")
    p.add_argument("--epochs", type=int, default=None, help="default: 100, or 2 with --check")
    p.add_argument("--fraction", type=float, default=None, help="default: 1.0, or 0.1 with --check")
    p.add_argument("--model", default="yolo11n.pt", help="starting weights")
    p.add_argument("--data", type=Path, default=ROOT / "datasets" / "lars_yolo" / "lars.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 = auto-fit to GPU memory")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="mixed-precision (FP16) training; --no-amp for FP32 (Pascal GPUs gain little from FP16)")
    p.add_argument("--name", default=None, help="default: 'check' or '<model>_lars'")
    args = p.parse_args()

    if args.epochs is None:
        args.epochs = 2 if args.check else 100
    if args.fraction is None:
        args.fraction = 0.1 if args.check else 1.0
    if args.name is None:
        args.name = "check" if args.check else f"{Path(args.model).stem}_lars"
    return args


def main() -> None:
    args = parse_args()

    # Windows: redirected stdout defaults to cp1252, which crashes on Ultralytics' progress bars.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from ultralytics import YOLO  # lazy: keeps --help fast

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=args.amp,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,  # existing name -> Ultralytics suffixes check2, check3, ...
    )
    print(f"\nTrained model: {model.trainer.best}")
    print(f"Graphs, metrics and preview images: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()

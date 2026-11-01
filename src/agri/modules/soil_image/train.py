"""Train soil_image. Usage: python -m agri.modules.soil_image.train [options]

Options (all optional; the defaults come from configs/soil_image.yaml):
  --epochs N          override model.params.epochs
  --max-per-class N   cap the TRAIN partition per class; test stays full size
  --batch-size N      override model.params.batch_size
  --learning-rate X   override model.params.learning_rate
  --freeze-epochs N   override model.params.freeze_backbone_epochs
  --num-workers N     DataLoader worker processes (default 0; >0 overlaps JPEG decode)
  --image-size N      square input override; recorded in the artefact and manifest
  --linear-probe      freeze the backbone, cache features once, train only the head
  --no-pretrained     skip ImageNet initialisation (also the offline fallback)
  --version V         registry version to write (default: config version)
  --no-register       train and measure without writing the registry

Why this is not a two-line wrapper around agri.common.runner.run_train: that helper
drives the tabular pipeline, whose loader raises for a directory dataset
("reads a directory, not a CSV; use its own loader"). The image module therefore
brings its own loader, in agri.modules.soil_image.training, and keeps the same
contract: JSON on stdout, a non-zero exit code when the metric gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys

from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging

from .training import train_soil_image

_log = get_logger("modules.soil_image.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the soil_image classifier")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument(
        "--max-per-class", type=int, default=None, help="cap training images per class (test stays full)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="override config batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="override config learning rate")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="override config freeze_backbone_epochs (train the head only for that many epochs)",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument(
        "--image-size", type=int, default=None, help="square input size override, recorded in the artefact"
    )
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help=(
            "freeze the backbone, cache its features once and train only the head "
            "(same transfer-learning recipe, far faster on CPU)"
        ),
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="do not initialise from ImageNet weights"
    )
    parser.add_argument("--version", default=None, help="registry version to write")
    parser.add_argument("--no-register", action="store_true", help="do not write to the registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    try:
        result = train_soil_image(
            epochs=args.epochs,
            max_per_class=args.max_per_class,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained=False if args.no_pretrained else None,
            freeze_epochs=args.freeze_epochs,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size) if args.image_size else None,
            linear_probe=args.linear_probe,
            register=not args.no_register,
            version=args.version,
        )
    except AgriError as exc:
        _log.error("training failed", extra={"module": "soil_image", "code": exc.code, "error": exc.message})
        print(
            json.dumps({"ok": False, "module": "soil_image", "error": exc.to_dict()}, indent=2, default=str)
        )
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

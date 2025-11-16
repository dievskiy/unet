#!/usr/bin/env python3
"""Convert a trained Lightning checkpoint into the ONNX artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import UnetConfig
from src.unet.trainer import UnetTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .ckpt file produced by PyTorch Lightning",
    )
    parser.add_argument(
        "--output",
        default="save_dir/model_unet.onnx",
        help="Destination ONNX file (defaults to save_dir/model_unet.onnx)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Training config JSON used to recover target size / channels",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version to export with (default: 17)",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> UnetConfig:
    cfg = Path(config_path)
    if cfg.exists():
        return UnetConfig.from_json(str(cfg))
    return UnetConfig()


def main() -> int:
    args = _parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = _load_config(args.config)

    model = UnetTrainer.load_from_checkpoint(
        str(ckpt_path),
        learning_rate=config.lr,
        num_classes=config.num_classes,
        ladder_size=config.ladder_size,
        map_location="cpu",
    )
    model.eval()

    dummy = torch.randn(1, config.num_classes, config.target_size[0], config.target_size[1])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.to_onnx(
        str(output_path),
        input_sample=dummy,
        export_params=True,
        opset_version=args.opset_version,
    )

    print(f"Exported {ckpt_path} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

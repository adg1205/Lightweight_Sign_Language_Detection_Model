import argparse
import json
from pathlib import Path

import torch

from train_lightweight_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unquantized PyTorch checkpoints to ONNX format."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="Root directory containing per-architecture model folders.",
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=["tiny_cnn", "mobilenetv2_025", "mcuformer_lite"],
        help="Architectures to export.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_model.pt",
        help="Checkpoint filename inside each architecture folder.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="model_fp32.onnx",
        help="Output ONNX filename inside each architecture folder.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def export_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path, opset: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = str(checkpoint["arch"])
    num_classes = int(checkpoint["num_classes"])
    img_size = int(checkpoint.get("img_size", 64))

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 1, img_size, img_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    meta = {
        "arch": arch,
        "num_classes": num_classes,
        "img_size": img_size,
        "checkpoint": str(checkpoint_path),
        "format": "ONNX fp32",
        "opset": opset,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    failures = []

    for arch in args.archs:
        checkpoint_path = args.models_root / arch / args.checkpoint_name
        output_path = args.models_root / arch / args.output_name

        if not checkpoint_path.exists():
            failures.append(f"Missing checkpoint: {checkpoint_path}")
            continue

        try:
            export_checkpoint_to_onnx(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                opset=args.opset,
            )
            size_bytes = output_path.stat().st_size
            print(f"Exported {arch}: {output_path} ({size_bytes / 1024:.2f} KB)")
        except Exception as exc:  # pragma: no cover
            failures.append(f"{arch}: {exc}")

    if failures:
        print("\nSome exports failed:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nAll requested unquantized models exported to ONNX successfully.")


if __name__ == "__main__":
    main()

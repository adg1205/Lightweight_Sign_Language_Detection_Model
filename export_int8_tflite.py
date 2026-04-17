import argparse
import json
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_lightweight_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained PyTorch model to INT8 using post-training quantization (PTQ)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/final_model.pt"),
        help="Path to trained PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--rep-data-dir",
        type=Path,
        default=Path("processed/train"),
        help="Directory for representative calibration images.",
    )
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-rep-samples", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model_int8.pt"),
        help="Output path for quantized TorchScript model.",
    )
    return parser.parse_args()


def make_rep_loader(rep_data_dir: Path, img_size: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(root=str(rep_data_dir), transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def calibrate(
    prepared_model: torch.nn.Module,
    rep_loader: DataLoader,
    max_samples: int,
) -> None:
    prepared_model.eval()
    seen = 0
    with torch.inference_mode():
        for images, _ in rep_loader:
            prepared_model(images)
            seen += images.size(0)
            if seen >= max_samples:
                break


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {args.model_path}")
    if not args.rep_data_dir.exists():
        raise RuntimeError(f"Representative data directory not found: {args.rep_data_dir}")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    arch = checkpoint["arch"]
    num_classes = int(checkpoint["num_classes"])
    img_size = int(checkpoint.get("img_size", args.img_size))

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # PTQ uses CPU backend for calibration and conversion.
    torch.backends.quantized.engine = "fbgemm"

    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(1, 1, img_size, img_size),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    rep_loader = make_rep_loader(
        rep_data_dir=args.rep_data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
    )
    calibrate(prepared, rep_loader, max_samples=args.num_rep_samples)

    quantized_model = convert_fx(prepared)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.script(quantized_model)
    scripted.save(str(args.output))

    meta_path = args.output.with_suffix(".json")
    meta = {
        "arch": arch,
        "num_classes": num_classes,
        "img_size": img_size,
        "quantization": "PTQ static INT8 (FX graph mode, fbgemm backend)",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_bytes = args.output.stat().st_size
    print(f"Saved INT8 quantized model to: {args.output}")
    print(f"Model size: {size_bytes} bytes ({size_bytes / 1024:.2f} KB)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

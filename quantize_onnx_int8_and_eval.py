import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ImageFolderCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        data_dir: Path,
        input_name: str,
        img_size: int,
        batch_size: int,
        num_samples: int,
    ) -> None:
        tfm = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
        ds = datasets.ImageFolder(root=str(data_dir), transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        self.input_name = input_name
        self.enum_data = self._load(loader, num_samples)

    def _load(
        self, loader: DataLoader, num_samples: int
    ) -> Iterator[Dict[str, np.ndarray]]:
        seen = 0
        for images, _ in loader:
            if seen >= num_samples:
                break
            batch = images.numpy().astype(np.float32)
            yield {self.input_name: batch}
            seen += batch.shape[0]

    def get_next(self) -> Dict[str, np.ndarray] | None:
        return next(self.enum_data, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8 with ONNX Runtime and evaluate accuracy."
    )
    parser.add_argument(
        "--onnx-fp32",
        type=Path,
        default=Path("models/mcuformer_lite/model_fp32.onnx"),
        help="Path to FP32 ONNX model.",
    )
    parser.add_argument(
        "--onnx-int8",
        type=Path,
        default=Path("models/mcuformer_lite/model_int8_ort.onnx"),
        help="Output path for INT8 ONNX model.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("processed/train"),
        help="Directory for calibration images (ImageFolder format).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("processed/val"),
        help="Directory for validation images (ImageFolder format).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("processed/test"),
        help="Directory for test images (flat folder).",
    )
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-calib-samples", type=int, default=500)
    return parser.parse_args()


def get_input_name(model_path: Path) -> str:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def quantize_model(
    fp32_path: Path,
    int8_path: Path,
    train_dir: Path,
    img_size: int,
    batch_size: int,
    num_calib_samples: int,
) -> None:
    input_name = get_input_name(fp32_path)
    data_reader = ImageFolderCalibrationDataReader(
        data_dir=train_dir,
        input_name=input_name,
        img_size=img_size,
        batch_size=batch_size,
        num_samples=num_calib_samples,
    )

    int8_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def make_val_loader(img_size: int, batch_size: int, val_dir: Path) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(root=str(val_dir), transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def eval_val_accuracy(model_path: Path, img_size: int, batch_size: int, val_dir: Path) -> float:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    loader = make_val_loader(img_size=img_size, batch_size=batch_size, val_dir=val_dir)

    total = 0
    correct = 0
    for images, labels in loader:
        batch = images.numpy().astype(np.float32)
        logits = sess.run(None, {input_name: batch})[0]
        preds = np.argmax(logits, axis=1)
        correct += int((preds == labels.numpy()).sum())
        total += labels.shape[0]

    return float(correct / total) if total else 0.0


def eval_test_accuracy(model_path: Path, img_size: int, test_dir: Path, class_names: List[str]) -> Tuple[float, int]:
    if not test_dir.exists():
        return 0.0, 0

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    files = sorted(
        [
            p
            for p in test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )

    total = 0
    correct = 0
    for p in files:
        label = p.stem.replace("_test", "")
        if label not in class_to_idx:
            continue
        image = tfm(datasets.folder.default_loader(str(p))).unsqueeze(0)
        logits = sess.run(None, {input_name: image.numpy().astype(np.float32)})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        correct += int(pred == class_to_idx[label])
        total += 1

    return (float(correct / total) if total else 0.0), total


def main() -> None:
    args = parse_args()

    if not args.onnx_fp32.exists():
        raise RuntimeError(f"Missing FP32 ONNX model: {args.onnx_fp32}")
    if not args.train_dir.exists():
        raise RuntimeError(f"Missing calibration directory: {args.train_dir}")
    if not args.val_dir.exists():
        raise RuntimeError(f"Missing validation directory: {args.val_dir}")

    class_names_path = args.onnx_fp32.parent / "class_names.json"
    if not class_names_path.exists():
        raise RuntimeError(f"Missing class_names.json: {class_names_path}")

    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

    quantize_model(
        fp32_path=args.onnx_fp32,
        int8_path=args.onnx_int8,
        train_dir=args.train_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_calib_samples=args.num_calib_samples,
    )

    val_acc = eval_val_accuracy(
        model_path=args.onnx_int8,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_dir=args.val_dir,
    )
    test_acc, test_samples = eval_test_accuracy(
        model_path=args.onnx_int8,
        img_size=args.img_size,
        test_dir=args.test_dir,
        class_names=class_names,
    )

    size_bytes = args.onnx_int8.stat().st_size
    metrics = {
        "arch": "mcuformer_lite",
        "img_size": int(args.img_size),
        "num_classes": int(len(class_names)),
        "val_accuracy_int8_onnx": float(val_acc),
        "test_accuracy_int8_onnx": float(test_acc),
        "test_samples": int(test_samples),
        "int8_onnx_size_bytes": int(size_bytes),
        "quant_tool": "onnxruntime.quantization.quantize_static",
        "quant_format": "QOperator",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "calib_samples": int(args.num_calib_samples),
    }

    metrics_path = args.onnx_int8.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

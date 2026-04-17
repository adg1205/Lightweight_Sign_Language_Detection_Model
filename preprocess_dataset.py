import argparse
import json
import random
from pathlib import Path
from typing import List

import tensorflow as tf

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize and grayscale ASL images, then create train/val split."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/asl_alphabet_train/asl_alphabet_train"),
        help="Input directory that contains one folder per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output root directory for processed dataset.",
    )
    parser.add_argument("--img-size", type=int, default=64, help="Output image size.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio per class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if no GPU is available.",
    )
    return parser.parse_args()


def setup_gpu(require_gpu: bool = False) -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if require_gpu:
            raise RuntimeError("No GPU found, but --require-gpu was set.")
        print("No GPU detected. Preprocessing will run on CPU.")
        return False

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Using GPU(s) for TensorFlow ops: {[gpu.name for gpu in gpus]}")
    return True


def list_class_dirs(input_dir: Path) -> List[Path]:
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    return sorted(class_dirs, key=lambda p: p.name)


def process_image(src_path: Path, dst_path: Path, img_size: int) -> None:
    raw = tf.io.read_file(str(src_path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)

    # Resize and grayscale conversion can execute on GPU when available.
    with tf.device("/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"):
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [img_size, img_size], method="bilinear")

    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    encoded = tf.io.encode_png(img)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tf.io.write_file(str(dst_path), encoded)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    setup_gpu(require_gpu=args.require_gpu)

    train_out = args.output_dir / "train"
    val_out = args.output_dir / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    class_dirs = list_class_dirs(args.input_dir)
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {args.input_dir}")

    class_names = [p.name for p in class_dirs]
    stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "img_size": args.img_size,
        "val_split": args.val_split,
        "num_classes": len(class_names),
        "classes": class_names,
        "per_class": {},
    }

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        if not files:
            continue

        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_split))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for src in train_files:
            dst = train_out / class_name / f"{src.stem}.png"
            process_image(src, dst, args.img_size)

        for src in val_files:
            dst = val_out / class_name / f"{src.stem}.png"
            process_image(src, dst, args.img_size)

        stats["per_class"][class_name] = {
            "original": len(files),
            "train": len(train_files),
            "val": len(val_files),
        }

    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (args.output_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"Processed classes: {len(class_names)}")
    print(f"Saved train set to: {train_out}")
    print(f"Saved val set to: {val_out}")
    print(f"Metadata: {args.output_dir / 'preprocess_stats.json'}")


if __name__ == "__main__":
    main()

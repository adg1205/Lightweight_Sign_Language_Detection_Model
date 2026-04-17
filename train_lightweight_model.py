import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class TinyASLCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(24, 24, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MobileNetV2Gray(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 0.25) -> None:
        super().__init__()
        self.model = models.mobilenet_v2(weights=None, width_mult=alpha)

        first_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN/MobileNet model for ASL classification (PyTorch)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("processed"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--arch",
        type=str,
        choices=["tiny_cnn", "mobilenetv2_025"],
        default="tiny_cnn",
        help="Model architecture.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"), help="Output model directory."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if no CUDA GPU is available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_gpu: bool = False) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        if require_gpu:
            raise RuntimeError("No CUDA GPU found, but --require-gpu was set.")
        print("No CUDA GPU detected. Training will run on CPU.")
        return torch.device("cpu")

    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "tiny_cnn":
        return TinyASLCNN(num_classes=num_classes)
    return MobileNetV2Gray(num_classes=num_classes, alpha=0.25)


def get_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def make_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, list]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            "Processed data not found. Run preprocess_dataset.py first to create train/val folders."
        )

    transform = get_transforms(img_size)

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=transform)

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError("Train/val class mappings differ. Re-run preprocessing.")

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    class_names = train_ds.classes
    return train_loader, val_loader, class_names


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if device.type == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                if device.type == "cuda" and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(require_gpu=args.require_gpu)
    train_loader, val_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    num_classes = len(class_names)
    model = build_model(args.arch, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.model_dir / "best_model.pt"
    final_path = args.model_dir / "final_model.pt"

    best_val_acc = -1.0
    patience = 5
    no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "class_names": class_names,
                },
                best_path,
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": args.arch,
            "num_classes": num_classes,
            "img_size": args.img_size,
            "class_names": class_names,
        },
        final_path,
    )

    (args.model_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )

    metrics = {
        "arch": args.arch,
        "img_size": args.img_size,
        "num_classes": num_classes,
        "val_loss": float(history["val_loss"][-1]),
        "val_accuracy": float(history["val_acc"][-1]),
        "best_val_accuracy": float(best_val_acc),
        "epochs_trained": len(history["train_loss"]),
        "params": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
    }
    (args.model_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print(f"Saved best model: {best_path}")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()

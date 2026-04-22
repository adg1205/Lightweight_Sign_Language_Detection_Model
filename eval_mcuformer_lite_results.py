from pathlib import Path
import json
import random
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_lightweight_model import build_model

MODEL_DIR = Path('models/mcuformer_lite')
BEST_PATH = MODEL_DIR / 'best_model.pt'
INT8_PATH = MODEL_DIR / 'model_int8.pt'
VAL_DIR = Path('processed/val')
TEST_DIR = Path('processed/test')


def make_val_loader(img_size=64, batch_size=128):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(root=str(VAL_DIR), transform=tfm)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def eval_float_val():
    ckpt = torch.load(BEST_PATH, map_location='cpu')
    arch = ckpt['arch']
    num_classes = int(ckpt['num_classes'])
    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds, loader = make_val_loader(img_size=int(ckpt.get('img_size', 64)))

    total = 0
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0, len(ds.classes), num_classes, int(ckpt.get('img_size', 64))


def eval_int8_val(img_size=64):
    model = torch.jit.load(str(INT8_PATH), map_location='cpu')
    model.eval()
    _, loader = make_val_loader(img_size=img_size)

    total = 0
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def eval_int8_test_flat(class_names, img_size=64):
    if not TEST_DIR.exists():
        return None, 0

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    model = torch.jit.load(str(INT8_PATH), map_location='cpu')
    model.eval()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    files = sorted([p for p in TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    total = 0
    correct = 0
    for p in files:
        lbl = p.stem.replace('_test', '')
        if lbl not in class_to_idx:
            continue
        x = tfm(datasets.folder.default_loader(str(p))).unsqueeze(0)
        with torch.inference_mode():
            pred = int(model(x).argmax(dim=1).item())
        correct += int(pred == class_to_idx[lbl])
        total += 1
    return (correct / total if total else 0.0), total


def estimate_peak_activation_kb(img_size=64, max_samples=64):
    model = torch.jit.load(str(INT8_PATH), map_location='cpu')
    model.eval()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(root=str(VAL_DIR), transform=tfm)

    idxs = list(range(len(ds)))
    random.seed(42)
    random.shuffle(idxs)
    idxs = idxs[:max_samples]

    peak_op_alloc = 0
    max_out_bytes = 0

    with torch.inference_mode():
        for i in idxs:
            x, _ = ds[i]
            x = x.unsqueeze(0)
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
                y = model(x)
            max_out_bytes = max(max_out_bytes, y.nelement() * y.element_size())
            for evt in prof.key_averages():
                peak_op_alloc = max(peak_op_alloc, evt.self_cpu_memory_usage)

    input_bytes = 1 * 1 * img_size * img_size * 4
    est_peak = input_bytes + peak_op_alloc + max_out_bytes
    return est_peak / 1024.0


def main():
    ckpt = torch.load(BEST_PATH, map_location='cpu')
    class_names = ckpt['class_names']
    float_val_acc, n_val_classes, n_classes, img_size = eval_float_val()
    int8_val_acc = eval_int8_val(img_size=img_size)
    int8_test_acc, test_samples = eval_int8_test_flat(class_names, img_size=img_size)

    float_size = BEST_PATH.stat().st_size
    int8_size = INT8_PATH.stat().st_size
    params = sum(v.numel() for v in ckpt['model_state'].values())
    peak_kb = estimate_peak_activation_kb(img_size=img_size)

    metrics = {
        'arch': ckpt['arch'],
        'img_size': img_size,
        'num_classes': n_classes,
        'val_accuracy_float': float(float_val_acc),
        'val_accuracy_int8': float(int8_val_acc),
        'val_accuracy_drop': float(float_val_acc - int8_val_acc),
        'test_accuracy_int8': (None if int8_test_acc is None else float(int8_test_acc)),
        'test_samples': int(test_samples),
        'params': int(params),
        'float_size_bytes': int(float_size),
        'int8_size_bytes': int(int8_size),
        'estimated_peak_activation_kb': float(peak_kb),
    }

    (MODEL_DIR / 'train_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (MODEL_DIR / 'class_names.json').write_text(json.dumps(class_names, indent=2), encoding='utf-8')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

from pathlib import Path
import json
import torch
from PIL import Image
from torchvision import transforms


def evaluate_flat_test(model_path, class_names_path, test_dir, img_size=64):
    class_names = json.loads(Path(class_names_path).read_text(encoding='utf-8'))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    model = torch.jit.load(str(model_path), map_location='cpu')
    model.eval()

    images = sorted([p for p in Path(test_dir).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])

    total = 0
    correct = 0
    skipped = 0

    with torch.inference_mode():
        for p in images:
            label = p.stem.replace('_test', '')
            if label not in class_to_idx:
                skipped += 1
                continue

            x = transform(Image.open(p).convert('RGB')).unsqueeze(0)
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            target = class_to_idx[label]

            correct += int(pred == target)
            total += 1

    acc = (correct / total) if total else 0.0
    return acc, total, skipped, len(images)


def main():
    test_root = Path('dataset/asl_alphabet_test/asl_alphabet_test')

    runs = [
        ('tiny_cnn_int8', Path('models/tiny_cnn/model_int8.pt'), Path('models/tiny_cnn/class_names.json')),
        ('mobilenetv2_025_int8', Path('models/mobilenetv2_025/model_int8.pt'), Path('models/mobilenetv2_025/class_names.json')),
    ]

    for name, model_path, class_names_path in runs:
        acc, used, skipped, all_files = evaluate_flat_test(
            model_path=model_path,
            class_names_path=class_names_path,
            test_dir=test_root,
            img_size=64,
        )
        print('{}: test_accuracy={:.6f} ({:.2f}%), evaluated_samples={}, skipped_samples={}, total_test_files={}'.format(
            name, acc, acc * 100.0, used, skipped, all_files
        ))


if __name__ == '__main__':
    main()

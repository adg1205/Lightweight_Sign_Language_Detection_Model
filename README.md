# Lightweight ASL Classifier for Microcontrollers

This project preprocesses the ASL alphabet dataset, trains a lightweight PyTorch image classifier for 29 classes, and exports an INT8 PyTorch model using post-training quantization (PTQ).

## 1) Install dependencies

```powershell
pip install -r requirements.txt
```

## 2) Preprocess dataset (64x64 grayscale)

Before preprocessing dataset, download the dataset from Kaggle using the link given: https://www.kaggle.com/datasets/grassknoted/asl-alphabet  
Rename the folder as dataset.

This creates:
- `processed/train/<class>/*.png`
- `processed/val/<class>/*.png`

```powershell
python preprocess_dataset.py --img-size 64 --val-split 0.1
```

To require GPU (fail if no GPU is detected):

```powershell
python preprocess_dataset.py --img-size 64 --val-split 0.1 --require-gpu
```

## 3) Train lightweight model (PyTorch)

### Option A: Tiny CNN (default, best for very small flash/RAM)

```powershell
python train_lightweight_model.py --arch tiny_cnn --epochs 20 --batch-size 64 --img-size 64
```

### Option B: MobileNetV2 alpha=0.25 (larger but still lightweight)

```powershell
python train_lightweight_model.py --arch mobilenetv2_025 --epochs 20 --batch-size 64 --img-size 64
```

To require GPU:

```powershell
python train_lightweight_model.py --arch tiny_cnn --epochs 20 --batch-size 64 --img-size 64 --require-gpu
```

Training artifacts are saved under `models/`:
- `final_model.pt`
- `best_model.pt`
- `class_names.json`
- `train_metrics.json`

## 4) Export INT8 model with PTQ (PyTorch)

```powershell
python export_int8_tflite.py --model-path models/final_model.pt --rep-data-dir processed/train --img-size 64 --num-rep-samples 500 --output models/model_int8.pt
```

This produces `models/model_int8.pt` (TorchScript) and prints its size in KB.

Note: PyTorch PTQ calibration and conversion are CPU-side steps. GPU is used for training.

## 5) Measured Results

The following values are from the runs in this repository (64x64 grayscale, 29 classes).

### Validation Accuracy (Before vs After PTQ)

| Model | Float best val accuracy | INT8 val accuracy | Accuracy drop |
|---|---:|---:|---:|
| Tiny CNN | 0.819310 (81.93%) | 0.713678 (71.37%) | 0.105632 (10.56%) |
| MobileNetV2 alpha=0.25 | 0.992644 (99.26%) | 0.983793 (98.38%) | 0.008851 (0.89%) |

### Test Accuracy (INT8 Models)

Test set format is a flat folder of 28 files (for example `A_test.jpg`).

| Model (INT8) | Test accuracy | Samples |
|---|---:|---:|
| Tiny CNN INT8 | 0.750000 (75.00%) | 28 |
| MobileNetV2 alpha=0.25 INT8 | 0.821429 (82.14%) | 28 |

### Model Size (Before vs After PTQ)

| Model | Float checkpoint size | INT8 checkpoint size |
|---|---:|---:|
| Tiny CNN | 24,098 bytes (23.53 KB) (`models/tiny_cnn/best_model.pt`) | 42,782 bytes (41.78 KB) (`models/tiny_cnn/model_int8.pt`) |
| MobileNetV2 alpha=0.25 | 1,253,624 bytes (1224.24 KB) (`models/mobilenetv2_025/best_model.pt`) | 485,510 bytes (474.13 KB) (`models/mobilenetv2_025/model_int8.pt`) |

Note: for very small models like Tiny CNN, the serialized TorchScript container and quantization metadata can outweigh weight-size savings, so the `.pt` file can become larger after PTQ even though arithmetic is INT8.

## Notes for 512 KB RAM / 1 MB Flash

- Use `tiny_cnn` first.
- Keep `img-size=64` and grayscale input.
- If model size is still high, reduce channels in `tiny_cnn` or train with `img-size=48`.
- Final on-device RAM usage depends on the runtime arena size and operator implementation, not only model file size.

# Mars Rover Terrain Segmentation

DeepLabV3+ (ResNet-50) to label every pixel of Curiosity rover images - soil, bedrock, sand, big rock.

AI4Mars - NASA JPL: 18,100 train + 322 expert test images. We used the merged MSL split (MER + MSL).

## Results - Local Test (200 images, 2 epochs, CPU)

Evaluated on **20 paired images** from MSL test split. 255 = unlabeled / rover / range>30m - ignored in loss and metrics.

| Class | IoU |
|---|---|
| soil | **0.8157** |
| bedrock | 0.0000 |
| sand | 0.0000 |
| big rock | 0.0000 |
| **mIoU** | **0.2039** |

Pixel Acc (ignoring 255) = **0.9063** - MISLEADING. 90% pixels are soil, so predicting only soil already gives 90%. mIoU is the honest metric.

> Paper reports higher mIoU on full 18k images + GPU + 15+ epochs. Our local test is only to prove the pipeline works.

**What happened:**
Model learned soil very well (0.8157) in just 2 epochs, but predicted everything as soil, so other 3 classes are 0. With 200 images and 2 epochs, the model never sees enough bedrock / sand / big rock examples. Full training on Colab (18k images, 15 epochs, class weighting) is needed to learn the rare classes.

[Prediction 0](results/pred_0.png)
[Prediction 1](results/pred_1.png)
[Prediction 2](results/pred_2.png)
*Left: Image, Middle: Ground Truth (black = 255 unlabeled), Right: Prediction (orange = soil). Model predicts soil everywhere - it has only learned the dominant class so far.*

Full per-class results in `results/per_class_iou.csv`.

## How it works
1. Image 256x256 -> DeepLabV3+ ResNet-50 encoder + ASPP
2. Loss = CrossEntropy with `ignore_index=255` - unlabeled / rover / range>30m pixels never count
3. Metrics also ignore 255 - same rule for loss and metric

## Quickstart - Local Test (Mac CPU, small subset)
```bash
git clone https://github.com/shiv2899/mars-terrain-segmentation.git
cd mars-terrain-segmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Put ~200 images in data/ai4mars/ (see data/README.md)
PYTHONPATH=src python src/train.py --epochs 2 --limit 200 --batch 2
PYTHONPATH=src python src/evaluate.py
PYTHONPATH=src python src/predict.py
```

## Full Train - Colab GPU (Recommended)
1. Upload `train_colab.ipynb` to Google Colab, Runtime -> T4 GPU
2. Run all cells - downloads, trains 15 epochs, evaluates
3. Download `checkpoints/deeplab_best.pth` + `results/` back

## Project Structure
```
mars-terrain-segmentation/
├── data/           # ignored, ai4mars/
├── results/        # per_class_iou.csv, pred_*.png
├── src/
│   ├── config.py
│   ├── dataset.py  # 255 handling + merged dataset pairing by F...M1 suffix
│   ├── model.py    # DeepLabV3+ ResNet-50
│   ├── train.py
│   ├── evaluate.py # per-class IoU + mIoU
│   └── predict.py  # overlay plots
├── train_colab.ipynb
├── LICENSE.md
└── requirements.txt
```

## Limitations / Next Improvement
- Local test only 200 images, 2 epochs - rare classes not learned
- Next: Colab full train 18k images, 15 epochs, class-weighted loss + augmentation
- Ground truth is sparse - most pixels are 255 unlabeled, only small regions labelled

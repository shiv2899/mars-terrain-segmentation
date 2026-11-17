# Mars Terrain Segmentation - AI4Mars
# Classes: 0=soil, 1=bedrock, 2=sand, 3=big rock, 255=unlabeled (IGNORE)
NUM_CLASSES=4
IGNORE_INDEX=255
CLASS_NAMES=["soil","bedrock","sand","big rock"]
IMG_SIZE=256  # Colab: 256, Local test: 256 same, memory safe
BATCH=8  # Colab T4: 8-16, Mac CPU: 2-4
EPOCHS=15
LR=1e-4
DATA_ROOT="data/ai4mars"

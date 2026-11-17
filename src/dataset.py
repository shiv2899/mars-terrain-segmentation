"""AI4Mars dataset - FIXED for merged dataset.
Images: NLA_397586928EDR_F0010008AUT_04096M1.JPG
Labels: NLA_397586934EDR_F0010008AUT_04096M1.png
Names differ in NLA number, so pair by suffix after EDR_ : F0010008AUT_04096M1
255 = no label / rover / range>30m - IGNORED in loss + metrics.
"""
import os, glob
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MarsDataset(Dataset):
    def __init__(self, root, split="train", size=256, limit=None):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        # index labels by suffix after EDR_
        label_map = {}
        for p in glob.glob(os.path.join(lbl_dir, "*.png")):
            base = os.path.basename(p)
            key = base.split("EDR_")[-1].replace(".png", "")
            label_map[key] = p
        pairs = []
        for img_p in sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.basename(img_p)
            key = base.split("EDR_")[-1].replace(".JPG", "").replace(".jpg", "")
            if key in label_map:
                pairs.append((img_p, label_map[key]))
        if limit:
            pairs = pairs[:limit]
        self.pairs = pairs
        self.size = size
        print(f"MarsDataset {split}: {len(self.pairs)} paired images (from {len(glob.glob(os.path.join(img_dir, '*')))} images)")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        img_p, lbl_p = self.pairs[i]
        img = Image.open(img_p).convert("RGB").resize((self.size, self.size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        lbl = np.array(Image.open(lbl_p).resize((self.size, self.size), Image.NEAREST))
        lbl = np.where((lbl > 3) & (lbl != 255), 255, lbl).astype(np.int64)
        return img, torch.from_numpy(lbl), os.path.basename(img_p)

"""Evaluate on expert test 322, report per-class IoU + mIoU + pixel acc (paper comparison only)."""
import torch, numpy as np
from torch.utils.data import DataLoader
from dataset import MarsDataset
from model import build_model
from train import compute_iou
import config, pandas as pd

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    ds=MarsDataset(config.DATA_ROOT,"test"); dl=DataLoader(ds,batch_size=8)
    model=build_model(); model.load_state_dict(torch.load("checkpoints/deeplab_best.pth",map_location="cpu")); model=model.to(device); model.eval()
    all_iou=np.zeros(4); n=0; correct=0; total=0
    with torch.no_grad():
        for img,lbl,_ in dl:
            pred=model(img.to(device))["out"].argmax(1).cpu()
            for i in range(len(lbl)):
                all_iou+=np.array(compute_iou(pred[i],lbl[i])); n+=1
                valid=lbl[i]!=255; correct+=((pred[i]==lbl[i])&valid).sum().item(); total+=valid.sum().item()
    per=all_iou/n; print(f"Test mIoU={per.mean():.4f} Pixel Acc (ignoring 255)={correct/total:.4f}")
    for c,name in enumerate(config.CLASS_NAMES): print(f"{name}: IoU={per[c]:.4f}")
    pd.DataFrame({"class":config.CLASS_NAMES,"IoU":per}).to_csv("results/per_class_iou.csv",index=False)
    print("Saved -> results/per_class_iou.csv")
if __name__=="__main__": main()

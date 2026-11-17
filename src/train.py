"""Train DeepLabV3+ ResNet-50. Local: --limit 200 --epochs 2 for test. Colab: full."""
import argparse, torch, torch.nn as nn, numpy as np, os
from torch.utils.data import DataLoader
from dataset import MarsDataset
from model import build_model
import config

def compute_iou(pred, target, num_classes=4, ignore=255):
    ious=[]
    for c in range(num_classes):
        pm=(pred==c); tm=(target==c)
        # ignore 255 pixels
        valid=(target!=ignore)
        pm=pm & valid; tm=tm & valid
        inter=(pm & tm).sum(); union=(pm | tm).sum()
        ious.append(float(inter/max(union,1)))
    return ious

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--limit",type=int,default=None); ap.add_argument("--batch",type=int,default=8)
    a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tr=MarsDataset(config.DATA_ROOT,"train",limit=a.limit); va=MarsDataset(config.DATA_ROOT,"test",limit=100 if a.limit else None)
    print(f"Train {len(tr)} Val {len(va)}")
    tr_dl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=2); va_dl=DataLoader(va,batch_size=a.batch)
    model=build_model().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn=nn.CrossEntropyLoss(ignore_index=255)
    best=0
    os.makedirs("checkpoints",exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tot=0
        for img,lbl,_ in tr_dl:
            img,lbl=img.to(device),lbl.to(device)
            opt.zero_grad(); out=model(img)["out"]; loss=loss_fn(out,lbl); loss.backward(); opt.step(); tot+=loss.item()
        # val mIoU
        model.eval(); all_iou=np.zeros(4); n=0
        with torch.no_grad():
            for img,lbl,_ in va_dl:
                pred=model(img.to(device))["out"].argmax(1).cpu()
                for i in range(len(lbl)): all_iou+=np.array(compute_iou(pred[i],lbl[i])); n+=1
        miou=(all_iou/n).mean()
        print(f"Epoch {ep+1}/{a.epochs} loss={tot/len(tr_dl):.4f} Val mIoU={miou:.4f} Per-class={np.round(all_iou/n,3)}")
        if miou>best: best=miou; torch.save(model.state_dict(),"checkpoints/deeplab_best.pth")
        sched.step()
    print(f"Best Val mIoU {best:.4f} saved to checkpoints/deeplab_best.pth")
if __name__=="__main__": main()

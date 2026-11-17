"""Save 3 overlay plots: image / ground truth / prediction. Usage: PYTHONPATH=src python src/predict.py --num 3"""
import torch, matplotlib.pyplot as plt, numpy as np, os
from dataset import MarsDataset
from model import build_model
import config
COLORS=np.array([[139,69,19],[128,128,128],[237,201,175],[255,0,0]],dtype=np.uint8)  # soil brown, bedrock gray, sand, big rock red
def colorize(mask):
    out=np.zeros((*mask.shape,3),dtype=np.uint8)
    for c in range(4): out[mask==c]=COLORS[c]
    out[mask==255]=[0,0,0]
    return out
def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    ds=MarsDataset(config.DATA_ROOT,"test"); model=build_model(); model.load_state_dict(torch.load("checkpoints/deeplab_best.pth",map_location="cpu")); model=model.to(device); model.eval()
    os.makedirs("results",exist_ok=True)
    for idx in range(3):
        img,lbl,name=ds[idx]
        with torch.no_grad(): pred=model(img.unsqueeze(0).to(device))["out"].argmax(1)[0].cpu().numpy()
        fig,ax=plt.subplots(1,3,figsize=(12,4))
        # denorm
        im=img.numpy().transpose(1,2,0)*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]); im=np.clip(im,0,1)
        ax[0].imshow(im); ax[0].set_title(f"Image {name}"); ax[1].imshow(colorize(lbl.numpy())); ax[1].set_title("Ground Truth"); ax[2].imshow(colorize(pred)); ax[2].set_title("Prediction")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(f"results/pred_{idx}.png",dpi=120); print(f"Saved results/pred_{idx}.png")
if __name__=="__main__": main()

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
def build_model(num_classes=4, pretrained=True):
    w=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    m=deeplabv3_resnet50(weights=w)
    m.classifier[4]=nn.Conv2d(256,num_classes,1)
    # aux classifier also must match
    if m.aux_classifier is not None:
        m.aux_classifier[4]=nn.Conv2d(256,num_classes,1)
    return m

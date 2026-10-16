"""MobileNetV3 Small with a 32-class head. Shared by train / evaluate / predict."""
import torch.nn as nn
from torchvision import models
from labels import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


def set_trainable(model, unfreeze_from):
    """Blocks model.features[i] with i < unfreeze_from are frozen. The classifier is always trainable.
    unfreeze_from = len(model.features) -> feature extractor fully frozen (head-only stage)."""
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= unfreeze_from
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_train_mode(model, unfreeze_from):
    """Frozen blocks stay in eval mode so their BatchNorm statistics are not changed."""
    model.train()
    for i, block in enumerate(model.features):
        if i < unfreeze_from:
            block.eval()

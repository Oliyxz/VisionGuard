import torch
import torchvision.models as models

def get_vit_model(pretrained=True):
    """
    Load Vision Transformer (ViT) model.
    """
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    return model

def get_convnext_model(pretrained=True):
    """
    Load ConvNeXt model.
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    return model

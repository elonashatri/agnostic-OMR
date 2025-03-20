"""Vision Transformer encoder for music notation."""

import torch.nn as nn
import timm
from config.config import Config

def create_vit_encoder(model_name="vit_small_patch16_224", pretrained=True):
    """
    Create a Vision Transformer encoder.
    
    Args:
        model_name: Name of the ViT model from timm
        pretrained: Whether to use pretrained weights
        
    Returns:
        ViT encoder model
    """
    # Create the ViT model
    model = timm.create_model(
        model_name, 
        pretrained=pretrained,
        img_size=Config.IMAGE_SIZE,  
        num_classes=0,  # Remove classification head
    )
    
    # Return the model
    return model
"""Transformer decoder for music notation generation."""

import torch
import torch.nn as nn

def create_transformer_decoder(d_model=384, nhead=6, num_decoder_layers=4, 
                              dim_feedforward=1024, dropout=0.1):
    """
    Create a Transformer decoder for music notation generation.
    
    Args:
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        
    Returns:
        Transformer decoder
    """
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    
    decoder = nn.TransformerDecoder(
        decoder_layer,
        num_layers=num_decoder_layers
    )
    
    return decoder
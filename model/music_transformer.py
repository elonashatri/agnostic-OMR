"""Music notation transformer model."""

import torch
import torch.nn as nn
import math
import timm

from config.config import Config
from model.encoder import create_vit_encoder
from model.decoder import create_transformer_decoder


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding."""
    
    def __init__(self, d_model, max_h=224, max_w=224):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encodings
        h_pos = torch.arange(max_h).unsqueeze(1).expand(-1, max_w).reshape(-1)
        w_pos = torch.arange(max_w).expand(max_h, -1).reshape(-1)
        
        # Interleave dimensions to ensure both h and w get equal representation
        pe = torch.zeros(max_h * max_w, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Position encoding for height
        pe[:, 0::4] = torch.sin(h_pos.unsqueeze(1) * div_term)
        pe[:, 1::4] = torch.cos(h_pos.unsqueeze(1) * div_term)
        
        # Position encoding for width
        pe[:, 2::4] = torch.sin(w_pos.unsqueeze(1) * div_term)
        pe[:, 3::4] = torch.cos(w_pos.unsqueeze(1) * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.reshape(max_h, max_w, d_model))
    
    def forward(self, x):
        """
        Add positional encoding to the input features.
        
        Args:
            x: Input tensor [batch_size, h, w, d_model]
            
        Returns:
            x with positional encoding added
        """
        batch_size, h, w, d = x.size()
        pe = self.pe[:h, :w, :]
        return x + pe

class MusicNotationTransformer(nn.Module):
    """Transformer model for music notation understanding and generation."""
    
    def __init__(self, config=None):
        """
        Initialize the music notation transformer.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        if config is None:
            config = Config
        
        self.config = config
        
        # Vision Transformer encoder
        self.encoder = create_vit_encoder(
            model_name=config.VIT_MODEL,
            pretrained=True
        )
        
        # Embedding dimension from the encoder
        encoder_dim = self.encoder.embed_dim
        
        # Decoder embedding layers
        self.symbol_embedding = nn.Embedding(
            config.NUM_SYMBOL_TYPES, 
            encoder_dim
        )
        self.position_embedding = nn.Embedding(
            config.MAX_SEQ_LENGTH, 
            encoder_dim
        )
        
        # Transformer decoder
        self.decoder = create_transformer_decoder(
            d_model=encoder_dim,
            nhead=config.NUM_HEADS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=encoder_dim * 4,
            dropout=config.DROPOUT
        )
        
        # Output projection layers
        self.symbol_classifier = nn.Linear(encoder_dim, config.NUM_SYMBOL_TYPES)
        self.position_regressor = nn.Linear(encoder_dim, 4)  # x, y, width, height
        self.staff_position_classifier = nn.Linear(encoder_dim, 11)  # -5 to +5 staff positions
        

    def forward(self, images, target_symbols=None, teacher_forcing_ratio=1.0):
        """
        Forward pass through the model.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            target_symbols: Target symbol sequence for training
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode images
        memory = self.encoder(images)  # This might be [batch_size, embed_dim]
        
        # Check and reshape memory if needed
        if memory.dim() == 2:
            # If memory is [batch_size, embed_dim], reshape to [batch_size, 1, embed_dim]
            memory = memory.unsqueeze(1)
        
        # Now make sure memory has the right sequence dimension
        # TransformerDecoder expects memory of shape [batch_size, seq_len, embed_dim]
        # If your memory has the wrong dimensions, reshape it
        if memory.dim() != 3:
            # If it's not 3D, something's wrong - let's force it to correct shape
            print(f"Warning: Unexpected memory shape: {memory.shape}. Reshaping...")
            embed_dim = self.encoder.embed_dim
            memory = memory.view(batch_size, -1, embed_dim)
        
        # Process target symbols
        if target_symbols is not None:
            # Handle different input formats
            if target_symbols.dim() == 1:
                # For patch-based 1D tensor [seq_len]
                seq_len = target_symbols.size(0)
                # Reshape to [1, seq_len]
                target_symbols = target_symbols.unsqueeze(0)
                
                # Embed target symbols
                symbol_embeddings = self.symbol_embedding(target_symbols)  # [1, seq_len, embed_dim]
                
                # Add positional embeddings
                pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
                pos_embeddings = self.position_embedding(pos)
                
                # Combine embeddings
                decoder_input = symbol_embeddings + pos_embeddings
                
                # For single sample, we might need to repeat memory to match batch dimension
                if memory.size(0) != target_symbols.size(0):
                    # If batch sizes don't match, handle it
                    if memory.size(0) == 1:
                        # Repeat memory to match target symbols batch size
                        memory = memory.expand(target_symbols.size(0), -1, -1)
                    elif target_symbols.size(0) == 1:
                        # Or use only the first memory if target is a single sample
                        memory = memory[:1]
                        
                # Decode - now both decoder_input and memory should be 3D tensors
                # with matching batch dimensions
                decoder_output = self.decoder(decoder_input, memory)
                
                # Project to outputs
                symbol_logits = self.symbol_classifier(decoder_output)
                position_preds = self.position_regressor(decoder_output)
                staff_position_logits = self.staff_position_classifier(decoder_output)
                
            else:
                # Standard case [batch_size, seq_len]
                seq_len = target_symbols.size(1)
                
                # Embed target symbols
                symbol_embeddings = self.symbol_embedding(target_symbols)  # [batch_size, seq_len, embed_dim]
                
                # Add positional embeddings
                pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                pos_embeddings = self.position_embedding(pos)
                
                # Combine embeddings
                decoder_input = symbol_embeddings + pos_embeddings
                
                # Decode
                decoder_output = self.decoder(decoder_input, memory)
                
                # Project to outputs
                symbol_logits = self.symbol_classifier(decoder_output)
                position_preds = self.position_regressor(decoder_output)
                staff_position_logits = self.staff_position_classifier(decoder_output)
                
        else:
            # Inference mode (autoregressive generation)
            # For simplicity, implement a fixed-length generation
            max_length = 10  # Adjust as needed
            
            # Initialize with start token
            current_symbol = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            
            outputs = []
            for t in range(max_length):
                # Embed current symbol
                symbol_embeddings = self.symbol_embedding(current_symbol)  # [batch_size, 1, embed_dim]
                
                # Add positional embedding
                pos = torch.full((batch_size, 1), t, dtype=torch.long, device=device)
                pos_embeddings = self.position_embedding(pos)
                
                # Combine embeddings
                decoder_input = symbol_embeddings + pos_embeddings
                
                # Decode
                decoder_output = self.decoder(decoder_input, memory)
                
                # Project to outputs
                symbol_logits = self.symbol_classifier(decoder_output)
                position_preds = self.position_regressor(decoder_output)
                staff_position_logits = self.staff_position_classifier(decoder_output)
                
                outputs.append({
                    'symbol_logits': symbol_logits,
                    'position_preds': position_preds,
                    'staff_position_logits': staff_position_logits
                })
                
                # Update current symbol (greedy decoding)
                current_symbol = symbol_logits.argmax(dim=-1)
            
            # Concatenate outputs
            symbol_logits = torch.cat([o['symbol_logits'] for o in outputs], dim=1)
            position_preds = torch.cat([o['position_preds'] for o in outputs], dim=1)
            staff_position_logits = torch.cat([o['staff_position_logits'] for o in outputs], dim=1)
        
        return {
            'symbol_logits': symbol_logits,
            'position_preds': position_preds,
            'staff_position_logits': staff_position_logits
        }
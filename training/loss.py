"""Loss functions for music notation transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicNotationLoss(nn.Module):
    """Combined loss for music notation prediction."""
    
    def __init__(self, symbol_weight=1.0, position_weight=1.0, staff_position_weight=1.0):
        """
        Initialize the loss function.
        
        Args:
            symbol_weight: Weight for symbol classification loss
            position_weight: Weight for position regression loss
            staff_position_weight: Weight for staff position classification loss
        """
        super().__init__()
        self.symbol_weight = symbol_weight
        self.position_weight = position_weight
        self.staff_position_weight = staff_position_weight
        
        # Symbol classification loss
        self.symbol_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Position regression loss
        self.position_criterion = nn.SmoothL1Loss()
        
        # Staff position classification loss
        self.staff_position_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, predictions, targets, mask=None):
        """
        Calculate the loss.
        
        Args:
            predictions: Dictionary of predictions from the model
            targets: Dictionary of target values
            mask: Optional mask for sequence padding
            
        Returns:
            Total loss and dictionary of individual losses
        """
        # Extract predictions
        symbol_logits = predictions['symbol_logits']
        position_preds = predictions['position_preds']
        staff_position_logits = predictions['staff_position_logits']
        
        # Extract targets
        symbol_targets = targets['symbol_ids']
        position_targets = targets['positions']
        staff_position_targets = targets['staff_positions']
        
        # Apply mask if provided
        if mask is not None:
            symbol_logits = symbol_logits * mask.unsqueeze(-1)
            position_preds = position_preds * mask.unsqueeze(-1)
            staff_position_logits = staff_position_logits * mask.unsqueeze(-1)
        
        # Calculate symbol classification loss
        symbol_loss = self.symbol_criterion(
            symbol_logits.reshape(-1, symbol_logits.size(-1)),
            symbol_targets.reshape(-1)
        )
        
        # Calculate position regression loss
        position_loss = self.position_criterion(position_preds, position_targets)
        
        # Calculate staff position classification loss
        staff_position_loss = self.staff_position_criterion(
            staff_position_logits.reshape(-1, staff_position_logits.size(-1)),
            staff_position_targets.reshape(-1)
        )
        
        # Calculate total loss
        total_loss = (
            self.symbol_weight * symbol_loss +
            self.position_weight * position_loss +
            self.staff_position_weight * staff_position_loss
        )
        
        # Return total loss and individual losses
        return total_loss, {
            'symbol_loss': symbol_loss.item(),
            'position_loss': position_loss.item(),
            'staff_position_loss': staff_position_loss.item(),
            'total_loss': total_loss.item()
        }
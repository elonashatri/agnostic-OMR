"""Metrics for evaluating music notation transformer."""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

def compute_metrics(model, data_loader, device):
    """
    Compute evaluation metrics for the model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_symbol_preds = []
    all_symbol_targets = []
    all_position_preds = []
    all_position_targets = []
    all_staff_position_preds = []
    all_staff_position_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing metrics"):
            # Move batch to device
            images = batch['image'].to(device)
            notation = {k: v.to(device) for k, v in batch['notation'].items()}
            
            # Forward pass
            outputs = model(images, notation['symbol_ids'], teacher_forcing_ratio=1.0)
            
            # Get predictions
            symbol_preds = outputs['symbol_logits'].argmax(dim=-1)
            position_preds = outputs['position_preds']
            staff_position_preds = outputs['staff_position_logits'].argmax(dim=-1)
            
            # Get targets
            symbol_targets = notation['symbol_ids']
            position_targets = notation['positions']
            staff_position_targets = notation['staff_positions']
            
            # Collect predictions and targets
            all_symbol_preds.append(symbol_preds.cpu().numpy())
            all_symbol_targets.append(symbol_targets.cpu().numpy())
            all_position_preds.append(position_preds.cpu().numpy())
            all_position_targets.append(position_targets.cpu().numpy())
            all_staff_position_preds.append(staff_position_preds.cpu().numpy())
            all_staff_position_targets.append(staff_position_targets.cpu().numpy())
    
    # Concatenate predictions and targets
    all_symbol_preds = np.concatenate(all_symbol_preds, axis=0)
    all_symbol_targets = np.concatenate(all_symbol_targets, axis=0)
    all_position_preds = np.concatenate(all_position_preds, axis=0)
    all_position_targets = np.concatenate(all_position_targets, axis=0)
    all_staff_position_preds = np.concatenate(all_staff_position_preds, axis=0)
    all_staff_position_targets = np.concatenate(all_staff_position_targets, axis=0)
    
    # Compute symbol accuracy
    symbol_accuracy = compute_symbol_accuracy(all_symbol_preds, all_symbol_targets)
    
    # Compute position error
    position_error = compute_position_error(all_position_preds, all_position_targets)
    
    # Compute staff position accuracy
    staff_position_accuracy = compute_staff_position_accuracy(
        all_staff_position_preds, 
        all_staff_position_targets
    )
    
    # Compute F1 score for symbol prediction
    symbol_f1 = compute_symbol_f1(all_symbol_preds, all_symbol_targets)
    
    # Compute combined score
    combined_score = 0.4 * symbol_accuracy + 0.3 * (1.0 - position_error / 100.0) + 0.3 * staff_position_accuracy
    
    return {
        'symbol_accuracy': symbol_accuracy,
        'position_error': position_error,
        'staff_position_accuracy': staff_position_accuracy,
        'symbol_f1': symbol_f1,
        'combined_score': combined_score
    }

def compute_symbol_accuracy(predictions, targets, ignore_index=-100):
    """
    Compute accuracy for symbol prediction.
    
    Args:
        predictions: Predicted symbol IDs
        targets: Target symbol IDs
        ignore_index: Index to ignore
        
    Returns:
        Accuracy score
    """
    # Flatten predictions and targets
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Create mask for valid indices
    mask = targets_flat != ignore_index
    
    # Compute accuracy
    accuracy = accuracy_score(targets_flat[mask], preds_flat[mask])
    
    return accuracy

def compute_position_error(predictions, targets):
    """
    Compute mean squared error for position prediction.
    
    Args:
        predictions: Predicted positions
        targets: Target positions
        
    Returns:
        Mean squared error
    """
    # Compute MSE
    mse = mean_squared_error(targets.reshape(-1, 4), predictions.reshape(-1, 4))
    
    # Scale for interpretability (0-100 range)
    scaled_error = min(100.0, mse * 100.0)
    
    return scaled_error

def compute_staff_position_accuracy(predictions, targets, ignore_index=-100):
    """
    Compute accuracy for staff position prediction.
    
    Args:
        predictions: Predicted staff positions
        targets: Target staff positions
        ignore_index: Index to ignore
        
    Returns:
        Accuracy score
    """
    # Flatten predictions and targets
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Create mask for valid indices
    mask = targets_flat != ignore_index
    
    # Compute accuracy
    accuracy = accuracy_score(targets_flat[mask], preds_flat[mask])
    
    return accuracy

def compute_symbol_f1(predictions, targets, ignore_index=-100):
    """
    Compute F1 score for symbol prediction.
    
    Args:
        predictions: Predicted symbol IDs
        targets: Target symbol IDs
        ignore_index: Index to ignore
        
    Returns:
        F1 score
    """
    # Flatten predictions and targets
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Create mask for valid indices
    mask = targets_flat != ignore_index
    
    # Compute F1 score (macro-averaged)
    f1 = f1_score(
        targets_flat[mask], 
        preds_flat[mask], 
        average='macro',
        zero_division=0
    )
    
    return f1

def compute_sequence_accuracy(predictions, targets, ignore_index=-100):
    """
    Compute sequence-level accuracy (exact match).
    
    Args:
        predictions: Predicted sequences
        targets: Target sequences
        ignore_index: Index to ignore
        
    Returns:
        Sequence accuracy score
    """
    # Create masks for valid indices
    masks = targets != ignore_index
    
    # Check for exact matches with masking
    correct = 0
    total = 0
    
    for pred, target, mask in zip(predictions, targets, masks):
        # Check if all valid positions match
        if np.all(pred[mask] == target[mask]):
            correct += 1
        total += 1
    
    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy
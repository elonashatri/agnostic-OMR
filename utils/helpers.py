"""Helper functions for music notation transformer."""

import os
import random
import logging
import json
import torch
import numpy as np
from datetime import datetime

def seed_everything(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_logging(output_dir):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('music_notation')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary
        is_best: Whether this is the best model so far
        filename: Checkpoint filename
        best_filename: Best model filename
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights
        optimizer: Optimizer to load state
        scheduler: Scheduler to load state
        device: Device to load checkpoint
        
    Returns:
        Loaded checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    """
    Count number of trainable parameters in model.
    
    Args:
        model: Model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_json(data, filename):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    """
    Load data from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded data
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_run_name(prefix='run'):
    """
    Generate a unique run name with timestamp.
    
    Args:
        prefix: Prefix for run name
        
    Returns:
        Run name
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"

def create_experiment_dir(base_dir='experiments', run_name=None):
    """
    Create experiment directory.
    
    Args:
        base_dir: Base directory
        run_name: Run name
        
    Returns:
        Experiment directory path
    """
    if run_name is None:
        run_name = generate_run_name()
    
    exp_dir = os.path.join(base_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    results_dir = os.path.join(exp_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return exp_dir

def calculate_position_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: First box [x, y, width, height]
        box2: Second box [x, y, width, height]
        
    Returns:
        IoU score
    """
    # Convert [x, y, width, height] to [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def convert_staff_position_to_midi(staff_position, clef_type='G', key_signature=None, octave_offset=4):
    """
    Convert staff position to MIDI note number.
    
    Args:
        staff_position: Staff position (can be float for spaces between lines)
        clef_type: Type of clef ('G', 'F', etc.)
        key_signature: Key signature (sharps/flats)
        octave_offset: Octave offset
        
    Returns:
        MIDI note number
    """
    # Base notes for different clefs (position 0)
    base_notes = {
        'G': 71,  # B4 (middle line in G clef)
        'F': 57,  # A3 (middle line in F clef)
        'C': 60,  # C4 (middle line in C clef)
    }
    
    # Get base note for the clef
    base_note = base_notes.get(clef_type, 71)  # Default to G clef if unknown
    
    # Each position is a semitone in a diatonic scale
    # Going up a line or space is +2 semitones (for most positions)
    midi_note = base_note + int(staff_position * 2)
    
    # Apply key signature adjustments if provided
    if key_signature is not None:
        # Implementation depends on your key signature representation
        # This is a simplified version
        pass
    
    return midi_note

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epochs=30):
    """
    Adjust learning rate based on epoch.
    
    Args:
        optimizer: Optimizer
        epoch: Current epoch
        initial_lr: Initial learning rate
        decay_rate: Learning rate decay rate
        decay_epochs: Epochs between learning rate decays
        
    Returns:
        New learning rate
    """
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def create_staff_position_encoding(max_positions=11):
    """
    Create one-hot encoding for staff positions.
    
    Args:
        max_positions: Maximum number of positions to encode (-5 to +5 = 11 positions)
        
    Returns:
        Dictionary mapping position values to one-hot encodings
    """
    encoding = {}
    for i in range(-max_positions//2, max_positions//2 + 1):
        one_hot = [0] * max_positions
        idx = i + max_positions//2
        one_hot[idx] = 1
        encoding[i] = one_hot
    return encoding
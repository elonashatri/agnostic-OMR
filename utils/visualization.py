"""Visualization utilities for music notation transformer."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import make_grid

def visualize_score_with_predictions(image, true_notation, pred_notation, score_width=2475, score_height=3504):
    """
    Visualize a music score with ground truth and predicted notation.
    
    Args:
        image: Input image
        true_notation: Ground truth notation
        pred_notation: Predicted notation
        score_width: Original score width
        score_height: Original score height
        
    Returns:
        Visualization image
    """
    # Convert to PIL Image if it's a numpy array or tensor
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4:  # Batch of images
            image = image[0]  # Take the first image
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype('uint8'))
    
    # Resize to original dimensions if needed
    if image.size != (score_width, score_height):
        image = image.resize((score_width, score_height))
    
    # Create a copy for drawing
    true_img = image.copy()
    pred_img = image.copy()
    
    # Create drawing contexts
    true_draw = ImageDraw.Draw(true_img)
    pred_draw = ImageDraw.Draw(pred_img)
    
    # Draw ground truth notation
    draw_notation(true_draw, true_notation, color=(0, 255, 0))  # Green for ground truth
    
    # Draw predicted notation
    draw_notation(pred_draw, pred_notation, color=(255, 0, 0))  # Red for predictions
    
    # Combine the images
    combined = np.concatenate([
        np.array(true_img),
        np.array(pred_img)
    ], axis=1)
    
    return combined

def draw_notation(draw, notation, color=(0, 255, 0)):
    """
    Draw notation elements on an image.
    
    Args:
        draw: PIL ImageDraw object
        notation: Notation data
        color: Color for drawing
    """
    for item in notation:
        # Extract position
        if 'position' in item:
            x, y, width, height = item['position']
            
            # Draw bounding box
            draw.rectangle(
                [(x, y), (x + width, y + height)],
                outline=color,
                width=2
            )
            
            # Draw symbol type
            if 'type' in item:
                symbol_type = item['type']
                draw.text((x, y - 15), symbol_type, fill=color)
            
            # Draw staff position if available
            if 'staff_position' in item:
                staff_pos = item['staff_position']
                draw.text((x, y + height + 5), staff_pos, fill=color)

def visualize_attention(image, attention_weights, head_idx=0, save_path=None):
    """
    Visualize attention weights from the transformer model.
    
    Args:
        image: Input image
        attention_weights: Attention weights from model
        head_idx: Attention head index to visualize
        save_path: Path to save visualization
        
    Returns:
        Attention visualization
    """
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:  # Batch of images
            image = image[0]  # Take the first image
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Extract attention for the specified head
    if isinstance(attention_weights, torch.Tensor):
        attention = attention_weights[0, head_idx].cpu().numpy()
    else:
        attention = attention_weights
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Show original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show attention heatmap
    im = ax2.imshow(attention, cmap='viridis')
    ax2.set_title(f'Attention Weights (Head {head_idx})')
    fig.colorbar(im, ax=ax2)
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return fig

def plot_training_progress(train_losses, val_losses, metrics=None, save_path=None):
    """
    Plot training and validation losses and metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of metrics to plot
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    if metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics if provided
    if metrics:
        for name, values in metrics.items():
            ax2.plot(epochs, values, label=name)
        ax2.set_title('Evaluation Metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def visualize_predictions_grid(images, true_notations, pred_notations, max_images=16, save_path=None):
    """
    Create a grid of images with predictions.
    
    Args:
        images: Batch of images
        true_notations: Ground truth notations
        pred_notations: Predicted notations
        max_images: Maximum number of images to show
        save_path: Path to save visualization
        
    Returns:
        Grid visualization
    """
    n_images = min(len(images), max_images)
    
    # Create a figure
    fig, axes = plt.subplots(n_images, 2, figsize=(14, n_images * 3))
    
    # If only one image, make sure axes is 2D
    if n_images == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(n_images):
        # Get image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Normalize if needed
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        
        # Get notations
        true_notation = true_notations[i]
        pred_notation = pred_notations[i]
        
        # Draw ground truth
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Ground Truth')
        draw_notation_on_axes(axes[i, 0], true_notation, color='g')
        
        # Draw prediction
        axes[i, 1].imshow(img)
        axes[i, 1].set_title('Prediction')
        draw_notation_on_axes(axes[i, 1], pred_notation, color='r')
        
        # Turn off axes
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def draw_notation_on_axes(ax, notation, color='g'):
    """
    Draw notation elements on matplotlib axes.
    
    Args:
        ax: Matplotlib axes
        notation: Notation data
        color: Color for drawing
    """
    for item in notation:
        # Extract position
        if 'position' in item:
            x, y, width, height = item['position']
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw symbol type
            if 'type' in item:
                symbol_type = item['type']
                ax.text(x, y - 5, symbol_type, color=color, fontsize=8)

def visualize_staff_lines(image, staff_lines, save_path=None):
    """
    Visualize detected staff lines on an image.
    
    Args:
        image: Input image
        staff_lines: List of staff line y-coordinates
        save_path: Path to save visualization
        
    Returns:
        Visualization image
    """
    # Convert to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:  # Batch of images
            image = image[0]  # Take the first image
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Create a copy for drawing
    img_with_lines = image.copy()
    
    # Convert to RGB if grayscale
    if len(img_with_lines.shape) == 2:
        img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_GRAY2RGB)
    elif img_with_lines.shape[2] == 1:
        img_with_lines = cv2.cvtColor(img_with_lines.squeeze(2), cv2.COLOR_GRAY2RGB)
    
    # Ensure image is in the right format
    if img_with_lines.max() <= 1.0:
        img_with_lines = (img_with_lines * 255).astype(np.uint8)
    
    # Draw each staff line
    for y in staff_lines:
        cv2.line(
            img_with_lines,
            (0, int(y)),
            (img_with_lines.shape[1], int(y)),
            (0, 0, 255),  # Red color
            2
        )
    
    # Save or return
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_with_lines, cv2.COLOR_RGB2BGR))
    
    return img_with_lines

def visualize_model_output(image, outputs, notation_vocab, save_path=None):
    """
    Visualize model outputs on an image.
    
    Args:
        image: Input image
        outputs: Model outputs (symbol logits, positions, etc.)
        notation_vocab: Vocabulary mapping IDs to symbol types
        save_path: Path to save visualization
        
    Returns:
        Visualization image
    """
    # Convert to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:  # Batch of images
            image = image[0]  # Take the first image
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Ensure image is in the right format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert to PIL for drawing
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    # Extract predictions
    symbol_preds = outputs['symbol_logits'].argmax(dim=-1)[0].cpu().numpy()
    position_preds = outputs['position_preds'][0].cpu().numpy()
    
    # Draw each prediction
    for i, (symbol_id, position) in enumerate(zip(symbol_preds, position_preds)):
        # Skip padding or end tokens
        if symbol_id == 0:  # Assuming 0 is padding
            continue
        
        # Get symbol type
        symbol_type = notation_vocab.get(symbol_id, f"Unknown-{symbol_id}")
        
        # Draw bounding box
        x, y, w, h = position
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=(255, 0, 0),
            width=2
        )
        
        # Draw symbol type
        draw.text((x, y - 15), symbol_type, fill=(255, 0, 0))
    
    # Save or return
    if save_path:
        pil_img.save(save_path)
    
    return np.array(pil_img)
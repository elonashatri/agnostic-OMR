"""Main training script for music notation transformer."""

import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime

from config.config import Config
from data.dataset import create_data_loaders
from model.music_transformer import MusicNotationTransformer
from training.trainer import Trainer
from training.metrics import compute_metrics
from utils.helpers import setup_logging, seed_everything

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train music notation transformer")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for data")
    parser.add_argument("--subset_size", type=int, default=None, help="Initial subset size")
    
    # Model arguments
    parser.add_argument("--vit_model", type=str, default=None, help="ViT model name")
    parser.add_argument("--decoder_layers", type=int, default=None, help="Number of decoder layers")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Hardware arguments
    parser.add_argument("--gpu_id", type=int, default=7, help="GPU ID to use (when multiple GPUs are available)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    
    # Checkpoint arguments
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=None, help="Epochs between checkpoint saves")
    
    # Validation arguments
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size")
    parser.add_argument("--val_interval", type=int, default=None, help="Epochs between validations")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training with args: {args}")
    
    # Set random seeds for reproducibility
    seed_everything(args.seed)
    
    # Create config with command line overrides
    config = Config()
    if args.data_root:
        config.DATA_ROOT = args.data_root
    if args.subset_size:
        config.INITIAL_SUBSET_SIZE = args.subset_size
    if args.vit_model:
        config.VIT_MODEL = args.vit_model
    if args.decoder_layers:
        config.NUM_DECODER_LAYERS = args.decoder_layers
    if args.hidden_dim:
        config.HIDDEN_DIM = args.hidden_dim
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.num_workers:
        config.NUM_WORKERS = args.num_workers
    if args.save_interval:
        config.SAVE_INTERVAL = args.save_interval
    if args.val_batch_size:
        config.VAL_BATCH_SIZE = args.val_batch_size
    if args.val_interval:
        config.VAL_INTERVAL = args.val_interval
    
    # Update device configuration based on GPU ID
    if args.device:
        config.DEVICE = args.device
    elif torch.cuda.is_available():
        # Set specific GPU if indicated
        if args.gpu_id >= 0 and args.gpu_id < torch.cuda.device_count():
            config.DEVICE = f"cuda:{args.gpu_id}"
            torch.cuda.set_device(args.gpu_id)
            logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            config.DEVICE = "cuda"
            logger.info(f"Using default GPU: {torch.cuda.get_device_name(0)}")
    else:
        config.DEVICE = "cpu"
        logger.info("CUDA not available, using CPU")
    
    # Set mixed precision flag
    config.MIXED_PRECISION = args.mixed_precision
    if config.MIXED_PRECISION:
        logger.info("Using mixed precision training")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Created data loaders with {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation samples")
    
    # Create model
    logger.info(f"Creating model ({config.VIT_MODEL})...")
    model = MusicNotationTransformer(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.global_step = checkpoint['global_step']
        logger.info(f"Resumed from epoch {start_epoch} with best val loss {trainer.best_val_loss:.4f}")
    
    # Train the model
    logger.info("Starting training...")
    trainer.current_epoch = start_epoch
    best_val_loss = trainer.train()
    logger.info(f"Training finished with best validation loss: {best_val_loss:.4f}")
    
    # Evaluate the final model
    logger.info("Evaluating final model...")
    model.eval()
    with torch.no_grad():
        metrics = compute_metrics(model, val_loader, config.DEVICE)
    
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save config used for training
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, 'w') as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Saved configuration to {config_path}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
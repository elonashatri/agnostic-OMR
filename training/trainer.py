"""Trainer for music notation transformer."""

import os
import time
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from training.loss import MusicNotationLoss
from utils.helpers import save_checkpoint, load_checkpoint

class Trainer:
    """Trainer for music notation transformer."""
    
    def __init__(self, model, train_loader, val_loader, config=None):
    
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
        """
        if config is None:
            config = Config
        
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Move model to device
        self.device = torch.device(config.DEVICE)
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Set up learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.EPOCHS
        )
        
        # Set up loss function
        self.criterion = MusicNotationLoss()
        
        # Set up gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Dataset expansion tracking
        self.expansion_schedule = config.EXPANSION_SCHEDULE
        self.next_expansion_idx = 0
        
        # Initialize TensorBoard writer
        self.log_dir = os.path.join(config.OUTPUT_DIR, 'tensorboard_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

    def train(self, epochs=None):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train for
        
        Returns:
            Best validation loss
        """
        if epochs is None:
            epochs = self.config.EPOCHS
        
        # Log model architecture and size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.writer.add_text('Model/Architecture', f"Total parameters: {total_params}, Trainable: {trainable_params}")
        
        # Log model hyperparameters
        hparam_dict = {
            'learning_rate': self.config.LEARNING_RATE,
            'batch_size': self.config.BATCH_SIZE,
            'weight_decay': self.config.WEIGHT_DECAY,
            'vit_model': self.config.VIT_MODEL,
            'hidden_dim': self.config.HIDDEN_DIM,
            'num_decoder_layers': self.config.NUM_DECODER_LAYERS,
            'num_heads': self.config.NUM_HEADS
        }
        self.writer.add_hparams(hparam_dict, {'metric/dummy': 0})
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.current_epoch = epoch
            
            # Check if we need to expand the dataset
            self._check_dataset_expansion()
            
            # Train for one epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.writer.add_scalar('BestValLoss', self.best_val_loss, self.current_epoch)
            
            save_checkpoint({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'global_step': self.global_step
            }, is_best, filename=f"checkpoint_epoch_{epoch}.pth")
            
            # Log epoch metrics to tensorboard
            self.writer.add_scalars('Loss/epoch', {
                'train': train_loss,
                'val': val_loss
            }, self.current_epoch)
            
            # Log current learning rate
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.best_val_loss
    
    def _train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        # Track individual loss components
        symbol_loss_total = 0
        position_loss_total = 0
        staff_position_loss_total = 0
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        # Set up gradient accumulation
        accum_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        
        for i, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            
            # Handle different notation formats (patch-based vs. standard)
            notation = {}
            for k, v in batch['notation'].items():
                notation[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            
            # For patch-based approach, we need to handle batch_indices
            is_patch_based = 'batch_indices' in notation
            
            try:
                # Forward pass with mixed precision
                with autocast():
                    # For patch-based approach, we might need additional handling
                    if is_patch_based and i == 0:  # Only print for first batch
                        print(f"Patch-based batch detected")
                        print(f"Image shape: {images.shape}")
                        print(f"Symbol IDs shape: {notation['symbol_ids'].shape}")
                        print(f"Batch indices shape: {notation['batch_indices'].shape}")
                        
                        # Log a sample image to tensorboard (first in batch)
                        if images.shape[0] > 0:
                            # Get first few images
                            sample_images = images[:min(4, images.shape[0])]
                            # Create grid of images
                            img_grid = torchvision.utils.make_grid(sample_images, normalize=True)
                            self.writer.add_image('Training/ImageBatch', img_grid, self.global_step)
                        
                    # Forward pass
                    outputs = self.model(images, notation['symbol_ids'], teacher_forcing_ratio=0.8)
                    
                    # Compute loss
                    total_loss, losses_dict = self.criterion(outputs, notation)
                    total_loss = total_loss / accum_steps  # Normalize for gradient accumulation
                
                # Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Update weights if gradient accumulation is complete
                if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Update progress bar
                epoch_loss += losses_dict['total_loss']
                symbol_loss_total += losses_dict.get('symbol_loss', 0)
                position_loss_total += losses_dict.get('position_loss', 0)
                staff_position_loss_total += losses_dict.get('staff_position_loss', 0)
                
                pbar.set_postfix(loss=f"{losses_dict['total_loss']:.4f}")
                
                # Log training metrics (every 10 batches or as appropriate)
                if i % 10 == 0:
                    self.writer.add_scalar('Loss/train_batch', losses_dict['total_loss'], self.global_step)
                    
                    # Log individual loss components if available
                    if 'symbol_loss' in losses_dict:
                        self.writer.add_scalar('Loss/train_symbol', losses_dict['symbol_loss'], self.global_step)
                    if 'position_loss' in losses_dict:
                        self.writer.add_scalar('Loss/train_position', losses_dict['position_loss'], self.global_step)
                    if 'staff_position_loss' in losses_dict:
                        self.writer.add_scalar('Loss/train_staff_position', losses_dict['staff_position_loss'], self.global_step)
                    
                    # Log current learning rate
                    self.writer.add_scalar('LearningRate/step', self.optimizer.param_groups[0]['lr'], self.global_step)
                
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Batch structure: {[k for k in batch.keys()]}")
                print(f"Notation keys: {[k for k in notation.keys()]}")
                if 'symbol_ids' in notation:
                    print(f"Symbol IDs shape: {notation['symbol_ids'].shape}")
                    print(f"Symbol IDs sample: {notation['symbol_ids'][:5]}")
                if 'positions' in notation:
                    print(f"Positions shape: {notation['positions'].shape}")
                
                # Log error to tensorboard
                self.writer.add_text('Errors/Train', f"Epoch {self.current_epoch}, Batch {i}: {str(e)}", self.global_step)
                
                # Skip this batch
                continue
            
            # Update global step
            self.global_step += 1
        
        # Calculate epoch averages
        num_batches = max(1, len(self.train_loader))
        avg_epoch_loss = epoch_loss / num_batches
        avg_symbol_loss = symbol_loss_total / num_batches
        avg_position_loss = position_loss_total / num_batches
        avg_staff_position_loss = staff_position_loss_total / num_batches
        
        # Log epoch average metrics
        self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, self.current_epoch)
        self.writer.add_scalar('Loss/train_symbol_epoch', avg_symbol_loss, self.current_epoch)
        self.writer.add_scalar('Loss/train_position_epoch', avg_position_loss, self.current_epoch)
        self.writer.add_scalar('Loss/train_staff_position_epoch', avg_staff_position_loss, self.current_epoch)
        
        return avg_epoch_loss
    
    def _validate(self):
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        # Track individual loss components
        symbol_loss_total = 0
        position_loss_total = 0
        staff_position_loss_total = 0
        
        # Create progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    images = batch['image'].to(self.device)
                    
                    # Handle different notation formats
                    notation = {}
                    for k, v in batch['notation'].items():
                        notation[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
                    
                    # Forward pass
                    outputs = self.model(images, notation['symbol_ids'], teacher_forcing_ratio=1.0)
                    
                    # Compute loss
                    total_loss, losses_dict = self.criterion(outputs, notation)
                    
                    # Update totals
                    val_loss += losses_dict['total_loss']
                    symbol_loss_total += losses_dict.get('symbol_loss', 0)
                    position_loss_total += losses_dict.get('position_loss', 0)
                    staff_position_loss_total += losses_dict.get('staff_position_loss', 0)
                    
                    # Update progress bar
                    pbar.set_postfix(loss=f"{losses_dict['total_loss']:.4f}")
                    num_batches += 1
                    
                    # Log sample validation images and predictions (every 5 epochs)
                    if self.current_epoch % 5 == 0 and batch_idx == 0:
                        # Get first few images
                        sample_images = images[:min(4, images.shape[0])]
                        # Create grid of images
                        img_grid = torchvision.utils.make_grid(sample_images, normalize=True)
                        self.writer.add_image('Validation/ImageBatch', img_grid, self.current_epoch)
                        
                        # Optionally log predictions visualization here if you have a visualization function
                    
                except Exception as e:
                    print(f"Error during validation: {e}")
                    # Log error to tensorboard
                    self.writer.add_text('Errors/Validation', f"Epoch {self.current_epoch}, Batch {batch_idx}: {str(e)}", self.current_epoch)
                    # Skip this batch
                    continue
        
        # Calculate validation averages
        num_batches = max(1, num_batches)
        avg_val_loss = val_loss / num_batches
        avg_symbol_loss = symbol_loss_total / num_batches
        avg_position_loss = position_loss_total / num_batches
        avg_staff_position_loss = staff_position_loss_total / num_batches
        
        # Log validation metrics
        self.writer.add_scalar('Loss/validation_epoch', avg_val_loss, self.current_epoch)
        self.writer.add_scalar('Loss/validation_symbol', avg_symbol_loss, self.current_epoch)
        self.writer.add_scalar('Loss/validation_position', avg_position_loss, self.current_epoch)
        self.writer.add_scalar('Loss/validation_staff_position', avg_staff_position_loss, self.current_epoch)
        
        return avg_val_loss
    
    def _check_dataset_expansion(self):
        """Check if we need to expand the dataset based on the current epoch."""
        if self.next_expansion_idx >= len(self.expansion_schedule):
            return
        
        target_epoch, new_size = self.expansion_schedule[self.next_expansion_idx]
        
        if self.current_epoch >= target_epoch:
            print(f"Expanding dataset to {new_size} samples...")
            # Log dataset expansion
            self.writer.add_text('Dataset/Expansion', f"Epoch {self.current_epoch}: Expanded to {new_size} samples", self.current_epoch)
            # This would be implemented based on your specific dataset expansion logic
            self.next_expansion_idx += 1
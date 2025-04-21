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
from utils.visualization import create_tensorboard_visualization, create_symbol_mapping


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
        self.id_to_symbol_map = None

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
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        # Initialize metric tracking
        metrics_totals = {
            'symbol_accuracy': 0.0,
            'position_accuracy': 0.0,
            'staff_position_accuracy': 0.0,
            'sequence_accuracy': 0.0
        }
        
        # Loss component tracking
        loss_components = {
            'total_loss': 0.0,
            'symbol_loss': 0.0,
            'position_loss': 0.0,
            'staff_position_loss': 0.0
        }
        
        # Create symbol mapping if not already available
        if not hasattr(self, 'id_to_symbol_map'):
            # Get the dataset from validation dataloader
            if isinstance(self.val_loader.dataset, torch.utils.data.Subset):
                dataset = self.val_loader.dataset.dataset
            else:
                dataset = self.val_loader.dataset
            
            # Create mapping from symbol IDs to names
            if hasattr(dataset, '_symbol_map'):
                # Invert the mapping (symbol->id becomes id->symbol)
                self.id_to_symbol_map = {v: k for k, v in dataset._symbol_map.items()}
                print(f"Created symbol map with {len(self.id_to_symbol_map)} entries")
            else:
                # Create default mapping
                self.id_to_symbol_map = {i: f"Symbol_{i}" for i in range(200)}
                print("Created default symbol map")
        
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
                    
                    # Print predictions for first batch only
                    if batch_idx == 0:
                        print("\n--------- SAMPLE PREDICTIONS ---------")
                        print(f"Symbol logits shape: {outputs['symbol_logits'].shape}")
                        pred_symbols = outputs['symbol_logits'].argmax(dim=-1)
                        print(f"Predicted symbols shape: {pred_symbols.shape}")
                        
                        # Take first 5 predictions or all if less than 5
                        num_to_show = min(5, pred_symbols.numel())
                        if pred_symbols.dim() > 1:
                            sample_preds = pred_symbols[0, :num_to_show]
                        else:
                            sample_preds = pred_symbols[:num_to_show]
                        
                        print(f"Sample predicted symbol IDs: {sample_preds.cpu().tolist()}")
                        
                        # Show symbol names if mapping exists
                        if self.id_to_symbol_map:
                            symbol_names = [self.id_to_symbol_map.get(s.item(), f"ID:{s.item()}") 
                                        for s in sample_preds]
                            print(f"Symbol names: {symbol_names}")
                        
                        # Show position predictions
                        print(f"Position predictions shape: {outputs['position_preds'].shape}")
                        if outputs['position_preds'].dim() > 2:
                            pos_preds = outputs['position_preds'][0, :num_to_show]
                        else:
                            pos_preds = outputs['position_preds'][:num_to_show]
                        
                        print(f"Sample position predictions: {pos_preds.cpu().tolist()}")
                        
                        # Show staff position predictions if available
                        if 'staff_position_logits' in outputs:
                            staff_preds = outputs['staff_position_logits'].argmax(dim=-1)
                            print(f"Staff position shape: {staff_preds.shape}")
                            if staff_preds.dim() > 1:
                                staff_samples = staff_preds[0, :num_to_show]
                            else:
                                staff_samples = staff_preds[:num_to_show]
                            print(f"Sample staff positions: {staff_samples.cpu().tolist()}")
                        
                        # Show some ground truth for comparison
                        print("\n--------- GROUND TRUTH ---------")
                        print(f"Target symbols shape: {notation['symbol_ids'].shape}")
                        print(f"Sample target symbol IDs: {notation['symbol_ids'][:num_to_show].cpu().tolist()}")
                        
                        if self.id_to_symbol_map:
                            gt_names = [self.id_to_symbol_map.get(s.item(), f"ID:{s.item()}") 
                                    for s in notation['symbol_ids'][:num_to_show]]
                            print(f"Ground truth symbol names: {gt_names}")
                        
                        print(f"Target positions shape: {notation['positions'].shape}")
                        print(f"Sample target positions: {notation['positions'][:num_to_show].cpu().tolist()}")
                        print("-------------------------------------\n")
                    
                    # Compute loss
                    total_loss, losses_dict = self.criterion(outputs, notation)
                    
                    # Rest of your validation code continues as before...
                    # Update loss totals
                    val_loss += losses_dict['total_loss']
                    for k, v in losses_dict.items():
                        if k in loss_components:
                            loss_components[k] += v
                    
                    # Update progress bar
                    pbar.set_postfix(loss=f"{losses_dict['total_loss']:.4f}")
                    
                    # Calculate detailed metrics
                    batch_metrics = self._calculate_batch_metrics(outputs, notation)
                    
                    # Update metric totals
                    for k, v in batch_metrics.items():
                        metrics_totals[k] += v
                    
                    # Update batch count
                    num_batches += 1
                    
                    # Create visualization
                    if batch_idx == 0:
                        try:
                            from utils.visualization import create_tensorboard_visualization
                            
                            viz_tensor = create_tensorboard_visualization(
                                images, outputs, notation, 
                                id_to_symbol_map=self.id_to_symbol_map
                            )
                            
                            if viz_tensor is not None:
                                self.writer.add_images(
                                    f'Validation/Predictions', 
                                    viz_tensor, 
                                    self.current_epoch
                                )
                        except Exception as viz_error:
                            print(f"Error creating visualization: {viz_error}")
                            import traceback
                            traceback.print_exc()
                    
                except Exception as e:
                    print(f"Error during validation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Log error to tensorboard
                    self.writer.add_text('Errors/Validation', 
                                        f"Epoch {self.current_epoch}, Batch {batch_idx}: {str(e)}", 
                                        self.current_epoch)
                    # Skip this batch
                    continue
            
            # Calculate averages
            num_batches = max(1, num_batches)
            avg_val_loss = val_loss / num_batches
            
            # Calculate metric averages
            metrics_avgs = {}
            for k, v in metrics_totals.items():
                metrics_avgs[k] = v / num_batches
            
            # Calculate loss component averages
            loss_avgs = {}
            for k, v in loss_components.items():
                loss_avgs[k] = v / num_batches
            
            # Log validation metrics and losses
            self.writer.add_scalar('Loss/validation_epoch', avg_val_loss, self.current_epoch)
            
            for k, v in loss_avgs.items():
                if k != 'total_loss':  # Already logged above
                    self.writer.add_scalar(f'Loss/validation_{k}', v, self.current_epoch)
            
            for k, v in metrics_avgs.items():
                self.writer.add_scalar(f'Metrics/{k}', v, self.current_epoch)
            
            # Replace dummy metric with real metrics in hparams (only once)
            if self.current_epoch == 0:
                hparam_dict = {
                    'learning_rate': self.config.LEARNING_RATE,
                    'batch_size': self.config.BATCH_SIZE,
                    'vit_model': self.config.VIT_MODEL,
                    'hidden_dim': self.config.HIDDEN_DIM,
                    'num_decoder_layers': self.config.NUM_DECODER_LAYERS
                }
                
                metrics_dict = {
                    'hparam/symbol_acc': metrics_avgs['symbol_accuracy'],
                    'hparam/position_acc': metrics_avgs['position_accuracy'],
                    'hparam/staff_position_acc': metrics_avgs.get('staff_position_accuracy', 0.0)
                }
                
                self.writer.add_hparams(hparam_dict, metrics_dict)
            
            return avg_val_loss

    def _calculate_batch_metrics(self, outputs, targets):
        """Calculate detailed metrics for a validation batch."""
        metrics = {}
        
        # Get predictions
        pred_symbols = outputs['symbol_logits'].argmax(dim=-1).cpu()
        pred_positions = outputs['position_preds'].cpu()
        
        if 'staff_position_logits' in outputs:
            pred_staff_positions = outputs['staff_position_logits'].argmax(dim=-1).cpu()
        else:
            pred_staff_positions = None
        
        # Get targets
        target_symbols = targets['symbol_ids'].cpu()
        target_positions = targets['positions'].cpu()
        
        if 'staff_positions' in targets:
            target_staff_positions = targets['staff_positions'].cpu()
        else:
            target_staff_positions = None
        
        # Flatten if needed
        if pred_symbols.dim() > 1:
            pred_symbols = pred_symbols.view(-1)
        if target_symbols.dim() > 1:
            target_symbols = target_symbols.view(-1)
        
        # Ensure compatible shapes for position data
        if pred_positions.dim() > 2:
            pred_positions = pred_positions.view(-1, pred_positions.size(-1))
        if target_positions.dim() > 2:
            target_positions = target_positions.view(-1, target_positions.size(-1))
        
        # 1. Symbol Accuracy
        correct_symbols = (pred_symbols == target_symbols).float().sum()
        total_symbols = target_symbols.numel()
        metrics['symbol_accuracy'] = (correct_symbols / total_symbols).item() if total_symbols > 0 else 0
        
        # 2. Position Accuracy (using IoU threshold of 0.5)
        position_correct = 0
        for pred_pos, target_pos in zip(pred_positions, target_positions):
            # Calculate IoU (Intersection over Union)
            x1_pred, y1_pred = pred_pos[0], pred_pos[1]
            w_pred, h_pred = pred_pos[2], pred_pos[3]
            x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
            
            x1_target, y1_target = target_pos[0], target_pos[1]
            w_target, h_target = target_pos[2], target_pos[3]
            x2_target, y2_target = x1_target + w_target, y1_target + h_target
            
            # Calculate intersection area
            x_inter1 = max(x1_pred, x1_target)
            y_inter1 = max(y1_pred, y1_target)
            x_inter2 = min(x2_pred, x2_target)
            y_inter2 = min(y2_pred, y2_target)
            
            width_inter = max(0, x_inter2 - x_inter1)
            height_inter = max(0, y_inter2 - y_inter1)
            area_inter = width_inter * height_inter
            
            # Calculate union area
            area_pred = w_pred * h_pred
            area_target = w_target * h_target
            area_union = area_pred + area_target - area_inter
            
            # Calculate IoU
            iou = area_inter / area_union if area_union > 0 else 0
            
            # Check if IoU exceeds threshold
            if iou >= 0.5:
                position_correct += 1
        
        metrics['position_accuracy'] = position_correct / total_symbols if total_symbols > 0 else 0
        
        # 3. Staff Position Accuracy
        if pred_staff_positions is not None and target_staff_positions is not None:
            # Ensure compatible shapes
            if pred_staff_positions.dim() > 1:
                pred_staff_positions = pred_staff_positions.view(-1)
            if target_staff_positions.dim() > 1:
                target_staff_positions = target_staff_positions.view(-1)
            
            correct_staff_positions = (pred_staff_positions == target_staff_positions).float().sum()
            metrics['staff_position_accuracy'] = (correct_staff_positions / total_symbols).item() if total_symbols > 0 else 0
        else:
            metrics['staff_position_accuracy'] = 0.0
        
        # 4. Sequence Accuracy (how well model preserves order)
        # This is more complex and depends on how you define sequence correctness
        # Here we use a simple metric: % of consecutive pairs that are correctly ordered
        sequence_correct = 0
        if total_symbols > 1:
            for i in range(total_symbols - 1):
                if (pred_symbols[i] == target_symbols[i] and 
                    pred_symbols[i+1] == target_symbols[i+1]):
                    sequence_correct += 1
            
            metrics['sequence_accuracy'] = sequence_correct / (total_symbols - 1) if total_symbols > 1 else 0
        else:
            metrics['sequence_accuracy'] = 0.0
        
        return metrics
    
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
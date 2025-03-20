"""Dataset implementations for music notation data."""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import cv2

from config.config import Config
from data.augmentation import get_train_transforms, get_val_transforms, staff_invariant_augmentation
from data.preprocessing import extract_notation_patches, create_relative_position_encoding

class MusicNotationDataset(Dataset):
    """Dataset for music notation images and their corresponding notation data."""
    
    def __init__(self, image_paths, notation_data, transform=None, config=None):
        """
        Args:
            image_paths: List of paths to images
            notation_data: List of notation data corresponding to images
            transform: Albumentations transformations to apply
            config: Configuration object with dataset options
        """
        self.image_paths = image_paths
        self.notation_data = notation_data
        self.transform = transform
        self.config = config if config else Config()
        
        # Create symbol map lazily when needed
        self._symbol_map = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Get notation data
        notation = self.notation_data[idx]
        
        # Check if using patch-based approach
        if hasattr(self.config, 'PATCH_BASED') and self.config.PATCH_BASED:
            return self._get_patch_based_item(image, notation, image_path)
        else:
            # Standard whole-image approach
            return self._get_whole_image_item(image, notation, image_path)
    
    def _get_whole_image_item(self, image, notation, image_path):
        """Process whole image approach."""
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor and normalize if not done in transform
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        # Process notation
        processed_notation = self._process_notation(notation)
        
        return {
            'image': image,
            'notation': processed_notation,
            'path': image_path
        }
    
    def _get_patch_based_item(self, image, notation, image_path):
        """Process patch-based approach."""
        # Use the function from preprocessing.py
        patch_size = getattr(self.config, 'PATCH_SIZE', (448, 224))
        context_factor = getattr(self.config, 'CONTEXT_FACTOR', 1.5)
        
        patches, positions, symbol_types = extract_notation_patches(
            image, 
            notation, 
            patch_size=patch_size,
            context_factor=context_factor
        )
        
        # Process each patch
        processed_patches = []
        for patch in patches:
            # Apply staff-invariant augmentation if configured and in training mode
            if getattr(self.config, 'STAFF_INVARIANT_AUGMENTATION', False) and hasattr(self, 'is_training') and self.is_training:
                patch = staff_invariant_augmentation(patch)
            
            # Apply transformations to each patch
            if self.transform:
                transformed = self.transform(image=patch)
                patch = transformed['image']
            
            # Convert to tensor if needed
            if not isinstance(patch, torch.Tensor):
                patch = torch.tensor(patch).permute(2, 0, 1).float() / 255.0
                
            processed_patches.append(patch)
        
        # Stack patches if any, otherwise create a dummy patch
        if processed_patches:
            patches_tensor = torch.stack(processed_patches)
        else:
            # Create a dummy patch with correct dimensions
            dummy = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.float32)
            if self.transform:
                transformed = self.transform(image=dummy)
                dummy = transformed['image']
            if not isinstance(dummy, torch.Tensor):
                dummy = torch.tensor(dummy).permute(2, 0, 1).float()
            patches_tensor = dummy.unsqueeze(0)
        
        # Process notation data for patches
        processed_notation = self._process_patch_notation(symbol_types, positions)
        
        return {
            'image': patches_tensor,  # Now contains multiple patches
            'notation': processed_notation,
            'path': image_path
        }
    
    def _process_notation(self, notation):
        """
        Process raw notation data into model-friendly format.
        
        Args:
            notation: Raw notation data for a single score
            
        Returns:
            Processed notation in a format suitable for the model
        """
        # This implementation depends on your specific notation format
        symbol_ids = []
        positions = []
        staff_positions = []
        unique_symbols = set()
        # Example processing for the format in your sample
        for item in notation:
            if isinstance(item, dict) and 'bounding_box' in item:
                # Skip system entries
                continue
                
            symbol_type = item.split('-')[0]
            unique_symbols.add(symbol_type)
            symbol_id = self._symbol_to_id(symbol_type)
            symbol_ids.append(symbol_id)
            
            # Extract position information if available
            pos_info = self._extract_position(item)
            positions.append(pos_info)
            
            # Extract staff position if available (L1, S2, etc.)
            staff_pos = self._extract_staff_position(item)
            staff_positions.append(staff_pos)
            
        print(f"Found unique symbols: {unique_symbols}")
        return {
            'symbol_ids': torch.tensor(symbol_ids, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float),
            'staff_positions': torch.tensor(staff_positions, dtype=torch.long)
        }
        
    
    def _process_patch_notation(self, symbol_types, positions):
        """
        Process patch-based notation data.
        
        Args:
            symbol_types: List of (symbol_type, staff_position) tuples
            positions: List of normalized [x, y, w, h] positions
            
        Returns:
            Processed notation in a format suitable for the model
        """
        symbol_ids = []
        staff_positions = []
        
        for symbol, staff_pos in symbol_types:
            # Convert symbol to ID
            symbol_id = self._symbol_to_id(symbol)
            symbol_ids.append(symbol_id)
            
            # Process staff position
            staff_position = self._extract_staff_position_from_str(staff_pos)
            staff_positions.append(staff_position)
        
        # Convert to tensors (handling empty case)
        if symbol_ids:
            symbol_ids = torch.tensor(symbol_ids, dtype=torch.long)
            positions_tensor = torch.tensor(positions, dtype=torch.float)
            staff_positions = torch.tensor(staff_positions, dtype=torch.long)
        else:
            symbol_ids = torch.zeros(1, dtype=torch.long)
            positions_tensor = torch.zeros((1, 4), dtype=torch.float)
            staff_positions = torch.zeros(1, dtype=torch.long)
        
        return {
            'symbol_ids': symbol_ids,
            'positions': positions_tensor,
            'staff_positions': staff_positions
        }
    
    def _symbol_to_id(self, symbol_type):
        """Convert symbol name to ID."""
        # Lazily create symbol map on first use
        if self._symbol_map is None:
            self._create_symbol_map()
        
        return self._symbol_map.get(symbol_type, len(self._symbol_map)-1)  # Unknown symbols get the last ID
    

    def _create_symbol_map(self):
        """Create a mapping from symbol names to IDs."""
        # Collect all unique symbol types
        symbol_types = set()
        
        print("Creating symbol map from notation data...")
        
        for i, notation in enumerate(self.notation_data):
            notation_symbols = set()
            for item in notation:
                if isinstance(item, dict) and 'bounding_box' in item:
                    continue
                
                if isinstance(item, str):
                    # Skip page index
                    if item.startswith('page_index'):
                        continue
                    
                    # Process individual elements
                    for element in item.split(';'):
                        element = element.strip()
                        if not element:
                            continue
                        
                        # Extract symbol type
                        if '-' in element:
                            try:
                                symbol_type = element.split('-')[0].strip()
                                notation_symbols.add(symbol_type)
                                symbol_types.add(symbol_type)
                            except Exception as e:
                                print(f"Error extracting symbol from: {element}")
                        else:
                            print(f"Skipping element without '-': {element}")
            
            # Print symbols found for each notation
            print(f"Notation {i}: Found {len(notation_symbols)} unique symbols: {notation_symbols}")
        
        print(f"Total unique symbols found: {len(symbol_types)}")
        for sym in sorted(symbol_types):
            print(f"  - {sym}")
        
        # Create mapping with 0 reserved for padding/unknown
        self._symbol_map = {'<pad>': 0}
        for i, symbol in enumerate(sorted(symbol_types)):
            self._symbol_map[symbol] = i + 1
        
        print(f"Created symbol map with {len(self._symbol_map)} entries")
        return self._symbol_map
    
    def _extract_position(self, item):
        """Extract position information from notation item."""
        # Parse the position information from your notation format
        position = [0, 0, 0, 0]  # [x, y, width, height]
        
        # Example extraction based on your format
        if "{'t': " in item:
            try:
                # Extract t, h, l, w values
                t = float(item.split("'t': ")[1].split(',')[0])
                h = float(item.split("'h': ")[1].split(',')[0])
                l = float(item.split("'l': ")[1].split(',')[0])
                w = float(item.split("'w': ")[1].split('}')[0])
                position = [l, t, w, h]
            except:
                pass
        
        return position
    
    def _extract_staff_position(self, item):
        """Extract staff position (L1, S2, etc.) from notation item."""
        # Parse the staff position from your notation format
        # Extract the string first
        staff_pos_str = self._extract_staff_position_str(item)
        
        # Then convert to numeric value
        return self._extract_staff_position_from_str(staff_pos_str)
    
    def _extract_staff_position_str(self, item):
        """Extract staff position string (L1, S2, etc.) from notation item."""
        staff_pos_str = None
        
        if "-L" in item or "-S" in item:
            try:
                parts = item.split('-')
                for part in parts:
                    if part.startswith('L') or part.startswith('S'):
                        # Get the staff position part without any following content
                        staff_pos_str = part.split('{')[0].strip()
                        break
            except:
                pass
        
        return staff_pos_str
    
    def _extract_staff_position_from_str(self, position_str):
        """Convert a staff position string (L1, S2, etc.) to a numeric value."""
        # Default: middle line (0)
        staff_pos = 0
        
        if position_str:
            try:
                type_char = position_str[0]
                number = int(position_str[1:])
                
                if type_char == 'L':
                    # Line, numbered from bottom to top
                    staff_pos = number - 3  # Map to -2, -1, 0, 1, 2
                elif type_char == 'S':
                    # Space, numbered from bottom to top
                    staff_pos = int((number - 2.5) * 2)  # Convert to integer representation
            except:
                pass
        
        return staff_pos

def patch_collate_fn(batch):
    """Custom collate function for patch-based processing."""
    # This handles batches where each item might have a different number of patches
    
    # Collect all paths
    paths = [item['path'] for item in batch]
    
    # Collect all patches from all items
    all_patches = []
    all_symbol_ids = []
    all_positions = []
    all_staff_positions = []
    batch_indices = []  # To track which batch item each patch belongs to
    
    for batch_idx, item in enumerate(batch):
        patches = item['image']  # Shape: [num_patches, C, H, W]
        num_patches = patches.shape[0]
        
        # Skip items with no valid patches
        if num_patches == 0:
            continue
        
        all_patches.append(patches)
        all_symbol_ids.append(item['notation']['symbol_ids'])
        all_positions.append(item['notation']['positions'])
        all_staff_positions.append(item['notation']['staff_positions'])
        
        # Record batch index for each patch
        batch_indices.extend([batch_idx] * num_patches)
    
    # Handle empty batch case
    if not all_patches:
        # Create a dummy batch
        dummy_patch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        dummy_symbol = torch.zeros((1,), dtype=torch.long)
        dummy_position = torch.zeros((1, 4), dtype=torch.float32)
        dummy_staff = torch.zeros((1,), dtype=torch.long)
        
        return {
            'image': dummy_patch,
            'notation': {
                'symbol_ids': dummy_symbol,
                'positions': dummy_position,
                'staff_positions': dummy_staff,
                'batch_indices': torch.zeros((1,), dtype=torch.long)
            },
            'path': paths
        }
    
    # Concatenate all tensors
    patches_tensor = torch.cat(all_patches, dim=0)
    symbol_ids_tensor = torch.cat(all_symbol_ids, dim=0)
    positions_tensor = torch.cat(all_positions, dim=0)
    staff_positions_tensor = torch.cat(all_staff_positions, dim=0)
    batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)
    
    return {
        'image': patches_tensor,
        'notation': {
            'symbol_ids': symbol_ids_tensor,
            'positions': positions_tensor,
            'staff_positions': staff_positions_tensor,
            'batch_indices': batch_indices_tensor  # Important for reconstruction
        },
        'path': paths
    }

def create_data_loaders(config=None):
    """Create train and validation data loaders."""
    if config is None:
        config = Config
    
    # Load image paths and notation data
    image_paths, notation_data = load_data(config.DATA_ROOT)
    
    # Create transforms
    train_transform = get_train_transforms(config.IMAGE_SIZE)
    val_transform = get_val_transforms(config.IMAGE_SIZE)
    
    # Create full dataset
    full_dataset = MusicNotationDataset(image_paths, notation_data, config=config)
    
    # Start with a subset if specified
    if config.INITIAL_SUBSET_SIZE > 0 and config.INITIAL_SUBSET_SIZE < len(full_dataset):
        subset_indices = np.random.choice(
            len(full_dataset), 
            config.INITIAL_SUBSET_SIZE, 
            replace=False
        )
        dataset = Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Set is_training flag for train dataset (used for augmentation)
    if hasattr(train_dataset, 'dataset'):
        train_dataset.dataset.is_training = True
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Get appropriate collate function based on config
    collate_fn = patch_collate_fn if getattr(config, 'PATCH_BASED', False) else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(config, 'NUM_WORKERS', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(config, 'VAL_BATCH_SIZE', config.BATCH_SIZE * 2),
        shuffle=False,
        num_workers=getattr(config, 'NUM_WORKERS', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def load_data(data_root):
    """
    Load image paths and notation data.
    
    Returns:
        image_paths: List of paths to images
        notation_data: List of notation data corresponding to images
    """
    # This would be replaced with your actual data loading logic
    image_paths = []
    notation_data = []
    
    # Example implementation
    images_dir = os.path.join(data_root, 'images')
    annotations_dir = os.path.join(data_root, 'annotations')
    
    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Warning: Data directories not found at {data_root}")
        return [], []
    
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(images_dir, filename)
            
            # Look for annotation files (.json or .agnostic)
            json_path = os.path.join(annotations_dir, filename.replace('.jpg', '.json').replace('.png', '.json'))
            agnostic_path = os.path.join(annotations_dir, filename.replace('.jpg', '.agnostic').replace('.png', '.agnostic'))
            
            annotation_path = None
            if os.path.exists(json_path):
                annotation_path = json_path
            elif os.path.exists(agnostic_path):
                annotation_path = agnostic_path
            
            if annotation_path:
                # Load based on file type
                if annotation_path.endswith('.json'):
                    with open(annotation_path, 'r') as f:
                        notation = json.load(f)
                else:  # .agnostic
                    with open(annotation_path, 'r') as f:
                        notation = f.read().splitlines()
                
                image_paths.append(image_path)
                notation_data.append(notation)
    
    print(f"Loaded {len(image_paths)} image-annotation pairs")
    return image_paths, notation_data
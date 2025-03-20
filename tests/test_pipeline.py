"""Test script for music notation transformer pipeline."""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import random
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_agnostic_file(file_path):
    """Read and parse agnostic file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    notation_data = []
    
    # Parse each line
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            notation_data.append(line)
    
    return notation_data

class MusicNotationDataset(torch.utils.data.Dataset):
    """Dataset for music notation images and their corresponding notation data."""
    
    def __init__(self, image_paths, notation_data, image_size=(224, 224)):
        """
        Args:
            image_paths: List of paths to images
            notation_data: List of notation data corresponding to images
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.notation_data = notation_data
        self.image_size = image_size
        self.symbol_map = self._create_symbol_map()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = self.preprocess_image(image_path)
        
        # Process notation data
        notation = self.notation_data[idx]
        processed_notation = self._process_notation(notation)
        
        return {
            'image': image,
            'notation': processed_notation
        }
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input."""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1)
        
        return image
    
    def _create_symbol_map(self):
        """Create a mapping from symbol types to IDs."""
        # Collect all unique symbol types
        symbol_types = set()
        for notations in self.notation_data:
            for item in notations:
                # Extract symbol type from notation
                if '-' in item and '{' in item:
                    symbol_type = item.split('-')[0].strip()
                    symbol_types.add(symbol_type)
        
        # Create mapping
        symbol_map = {symbol: idx + 1 for idx, symbol in enumerate(sorted(symbol_types))}
        symbol_map['<pad>'] = 0  # Add padding token
        
        return symbol_map
    
    def _process_notation(self, notation):
        """Process notation data into tensors."""
        # List to store symbols and positions
        symbol_ids = []
        positions = []  # [x, y, width, height]
        
        for item in notation:
            # Skip page index and system entries
            if item.startswith('page_index') or (item.startswith('S-') and 'bounding_box' in item):
                continue
            
            # Process individual elements
            for element in item.split(';'):
                element = element.strip()
                if not element:
                    continue
                
                # Extract symbol type
                if '-' in element and '{' in element:
                    symbol_type = element.split('-')[0].strip()
                    symbol_id = self.symbol_map.get(symbol_type, 0)
                    
                    # Extract position info
                    if '{' in element and '}' in element:
                        pos_str = element[element.find('{'):element.find('}')+1]
                        try:
                            # Parse position
                            pos_dict = eval(pos_str.replace("'", '"'))
                            position = [
                                float(pos_dict.get('l', 0)) / 2475,  # Normalize to 0-1
                                float(pos_dict.get('t', 0)) / 3504,
                                float(pos_dict.get('w', 0)) / 2475,
                                float(pos_dict.get('h', 0)) / 3504,
                            ]
                            
                            symbol_ids.append(symbol_id)
                            positions.append(position)
                        except:
                            print(f"Failed to parse position in element: {element}")
        
        # Convert to tensors
        if symbol_ids:
            symbol_ids = torch.tensor(symbol_ids, dtype=torch.long)
            positions = torch.tensor(positions, dtype=torch.float)
        else:
            # Handle empty case
            symbol_ids = torch.zeros(1, dtype=torch.long)
            positions = torch.zeros((1, 4), dtype=torch.float)
        
        return {
            'symbol_ids': symbol_ids,
            'positions': positions
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    # Extract images and notations
    images = [item['image'] for item in batch]
    notations = [item['notation'] for item in batch]
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Pad sequences
    symbol_ids = [n['symbol_ids'] for n in notations]
    positions = [n['positions'] for n in notations]
    
    # Get lengths
    lengths = torch.tensor([len(ids) for ids in symbol_ids], dtype=torch.long)
    
    # Pad sequences
    padded_ids = pad_sequence(symbol_ids, batch_first=True, padding_value=0)
    padded_positions = pad_sequence(positions, batch_first=True, padding_value=0.0)
    
    return {
        'image': images,
        'notation': {
            'symbol_ids': padded_ids,
            'positions': padded_positions,
            'lengths': lengths
        }
    }

class SimpleTransformerModel(torch.nn.Module):
    """Simplified transformer model for testing."""
    
    def __init__(self, vocab_size=100, hidden_dim=256):
        super().__init__()
        # Vision encoder (simplified)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, hidden_dim, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        
        # Embedding
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        
        # Decoder (simplified)
        self.decoder = torch.nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )
        
        # Output projections
        self.symbol_classifier = torch.nn.Linear(hidden_dim, vocab_size)
        self.position_regressor = torch.nn.Linear(hidden_dim, 4)
    
    def forward(self, images, target_symbols=None):
        batch_size = images.shape[0]
        
        # Encode images
        features = self.cnn(images).view(batch_size, -1)
        memory = features.unsqueeze(1).repeat(1, 10, 1)  # Simplified memory
        
        # Decode (simplified)
        if target_symbols is not None:
            # Use target symbols for training
            embedded = self.embedding(target_symbols)
            decoder_output = self.decoder(embedded, memory)
        else:
            # Generate sequence for inference (simplified)
            decoder_output = memory
        
        # Project to outputs
        symbol_logits = self.symbol_classifier(decoder_output)
        position_preds = self.position_regressor(decoder_output)
        
        return {
            'symbol_logits': symbol_logits,
            'position_preds': position_preds
        }

def test_dataset():
    """Test dataset creation and loading."""
    # Config for test
    data_root = "/homes/es314/agnostic-OMR/data"
    batch_size = 2
    
    # Get file paths
    image_dir = os.path.join(data_root, "images")
    annotation_dir = os.path.join(data_root, "annotations")
    
    # Find image and annotation pairs
    image_paths = []
    notation_data = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            annotation_path = os.path.join(
                annotation_dir, 
                filename.replace('.png', '.agnostic')
            )
            
            if os.path.exists(annotation_path):
                print(f"Found pair: {filename} and {os.path.basename(annotation_path)}")
                image_paths.append(image_path)
                
                # Read and parse agnostic file
                notations = read_agnostic_file(annotation_path)
                notation_data.append(notations)
    
    print(f"Found {len(image_paths)} image-annotation pairs")
    
    # Test loading and preprocessing one image
    if image_paths:
        # Load image
        test_image_path = image_paths[0]
        test_image = Image.open(test_image_path).convert('RGB')
        print(f"Original image size: {test_image.size}")
        
        # Create dataset
        test_dataset = MusicNotationDataset(
            image_paths[:3], 
            notation_data[:3]
        )
        print(f"Created dataset with {len(test_dataset)} samples")
        
        # Test one sample
        sample = test_dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample notation symbols: {sample['notation']['symbol_ids'].shape}")
        print(f"Sample notation positions: {sample['notation']['positions'].shape}")
        
        # Visualize processed image
        plt.figure(figsize=(8, 8))
        plt.imshow(sample['image'].permute(1, 2, 0).numpy())
        plt.title("Processed Image Sample")
        plt.savefig("test_processed_image.png")
        print(f"Saved processed image visualization to test_processed_image.png")
        
        # Test dataloader with custom collate
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Load a batch
        batch = next(iter(test_loader))
        print("\nBatch information:")
        print(f"- Image batch shape: {batch['image'].shape}")
        print(f"- Symbol IDs shape: {batch['notation']['symbol_ids'].shape}")
        print(f"- Positions shape: {batch['notation']['positions'].shape}")
        print(f"- Lengths: {batch['notation']['lengths']}")
        
        return test_loader
    else:
        print("No image-annotation pairs found!")
        return None

def test_model(test_loader):
    """Test model with a forward pass."""
    # Create a small model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get vocabulary size from the dataset
    vocab_size = max(test_loader.dataset.symbol_map.values()) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = SimpleTransformerModel(vocab_size=vocab_size)
    model = model.to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get a batch
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    target_symbols = batch['notation']['symbol_ids'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, target_symbols)
    
    print("\nModel output information:")
    print(f"- Symbol logits shape: {outputs['symbol_logits'].shape}")
    print(f"- Position predictions shape: {outputs['position_preds'].shape}")
    
    # Verify shapes match
    assert outputs['symbol_logits'].shape == target_symbols.shape + (vocab_size,)
    
    return model, outputs

def visualize_results(batch, outputs):
    """Visualize model predictions."""
    # Get first image from batch
    image = batch['image'][0].permute(1, 2, 0).numpy()
    
    # Get ground truth and predictions
    true_symbols = batch['notation']['symbol_ids'][0].numpy()
    true_positions = batch['notation']['positions'][0].numpy()
    
    pred_symbols = outputs['symbol_logits'][0].argmax(dim=-1).cpu().numpy()
    pred_positions = outputs['position_preds'][0].cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.imshow(image)
    
    # Draw ground truth (green)
    for symbol, pos in zip(true_symbols, true_positions):
        if symbol == 0:  # Skip padding
            continue
        x, y, w, h = pos
        # Convert normalized to pixel coordinates
        x, y = x * image.shape[1], y * image.shape[0]
        w, h = w * image.shape[1], h * image.shape[0]
        # Draw rectangle
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)
    
    # Draw predictions (red)
    for symbol, pos in zip(pred_symbols, pred_positions):
        if symbol == 0:  # Skip padding
            continue
        x, y, w, h = pos
        # Convert normalized to pixel coordinates
        x, y = x * image.shape[1], y * image.shape[0]
        w, h = w * image.shape[1], h * image.shape[0]
        # Draw rectangle
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.title("Model Predictions (Green=Ground Truth, Red=Prediction)")
    plt.axis('off')
    plt.savefig("test_predictions.png")
    print("Saved prediction visualization to test_predictions.png")

def main():
    """Run pipeline test."""
    print("Testing music notation transformer pipeline...")
    
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Test dataset and dataloader
    test_loader = test_dataset()
    
    if test_loader:
        # Test model
        model, outputs = test_model(test_loader)
        
        # Visualize results
        batch = next(iter(test_loader))
        visualize_results(batch, outputs)
        
        print("\nPipeline test complete!")

if __name__ == "__main__":
    main()
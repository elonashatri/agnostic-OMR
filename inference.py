# save as check_predictions.py
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model.music_transformer import MusicNotationTransformer
from config.config import Config
from data.dataset import MusicNotationDataset, load_data
from data.augmentation import get_val_transforms

def main():
    # Set up configuration
    config = Config()
    config.PATCH_BASED = True
    config.PATCH_SIZE = (224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint_path = "/import/c4dm-05/elona/agnostic-OMR/model_best.pth"
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = MusicNotationTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Load data
    data_root = config.DATA_ROOT
    print(f"Loading data from {data_root}")
    
    image_paths, notation_data = load_data(data_root)
    
    if not image_paths:
        print("No data found! Check your data root path.")
        return
    
    print(f"Loaded {len(image_paths)} samples")
    
    # Create dataset
    transform = get_val_transforms(config.IMAGE_SIZE)
    dataset = MusicNotationDataset(image_paths, notation_data, transform=transform, config=config)
    
    # Get symbol map for reference
    if not hasattr(dataset, '_symbol_map') or dataset._symbol_map is None:
        dataset._create_symbol_map()
    id_to_symbol = {v: k for k, v in dataset._symbol_map.items()}
    
    # Load a sample (let's take the first one)
    sample_idx = 0
    sample = dataset[sample_idx]
    
    # Extract patches and notation
    patches = sample['image']
    notation = sample['notation']
    
    print(f"Loaded sample with {patches.shape[0]} patches")
    print(f"Sample path: {sample['path']}")
    
    # Run inference - but handle each patch individually for patch-based approach
    predictions = []
    
    with torch.no_grad():
        # Process patches in small batches to avoid memory issues
        batch_size = 4
        num_patches = patches.shape[0]
        
        for i in range(0, num_patches, batch_size):
            # Get batch of patches
            end_idx = min(i + batch_size, num_patches)
            batch_patches = patches[i:end_idx].to(device)
            
            # Process each patch individually (key fix)
            for j in range(batch_patches.shape[0]):
                single_patch = batch_patches[j:j+1]  # Keep batch dimension [1, 3, 224, 224]
                # Create dummy target (not used with teacher_forcing_ratio=0)
                dummy_target = torch.zeros(1, 1, dtype=torch.long).to(device)
                
                # Run model
                out = model(single_patch, dummy_target, teacher_forcing_ratio=0.0)
                
                # Collect predictions
                pred = {
                    'symbol_id': out['symbol_logits'][0].argmax(-1).item(),
                    'position': out['position_preds'][0].cpu().numpy()
                }
                
                if 'staff_position_logits' in out:
                    pred['staff_position'] = out['staff_position_logits'][0].argmax(-1).item()
                
                predictions.append(pred)
    
    # Print predictions for a few patches
    print("\n----- PREDICTION RESULTS -----")
    num_to_show = min(5, num_patches)
    
    for i in range(num_to_show):
        # Get predictions for this patch
        pred = predictions[i]
        symbol_id = pred['symbol_id']
        symbol_name = id_to_symbol.get(symbol_id, f"Unknown-{symbol_id}")
        
        # Get ground truth
        gt_symbol_id = notation['symbol_ids'][i].item()
        gt_symbol_name = id_to_symbol.get(gt_symbol_id, f"Unknown-{gt_symbol_id}")
        
        # Get positions
        pos = pred['position']
        gt_pos = notation['positions'][i].cpu().numpy()
        
        # Print comparison
        print(f"\nPatch {i+1}:")
        print(f"  Predicted: {symbol_name}, Position: {pos}")
        print(f"  Ground Truth: {gt_symbol_name}, Position: {gt_pos}")
        
        # Print staff position if available
        if 'staff_position' in pred:
            staff_id = pred['staff_position']
            gt_staff_id = notation['staff_positions'][i].item()
            print(f"  Predicted Staff Position: {staff_id}")
            print(f"  Ground Truth Staff Position: {gt_staff_id}")
    
    # Visualize a few patches with predictions
    fig, axs = plt.subplots(2, num_to_show, figsize=(15, 6))
    
    for i in range(num_to_show):
        # Get the patch
        patch = patches[i].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize for visualization
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        
        # Get predictions
        pred = predictions[i]
        symbol_id = pred['symbol_id']
        symbol_name = id_to_symbol.get(symbol_id, f"Unknown-{symbol_id}")
        
        gt_symbol_id = notation['symbol_ids'][i].item()
        gt_symbol_name = id_to_symbol.get(gt_symbol_id, f"Unknown-{gt_symbol_id}")
        
        # Display patch and predictions
        axs[0, i].imshow(patch)
        axs[0, i].set_title(f"Pred: {symbol_name}")
        axs[0, i].axis('off')
        
        axs[1, i].imshow(patch)
        axs[1, i].set_title(f"GT: {gt_symbol_name}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print(f"\nSaved visualization to sample_predictions.png")

if __name__ == "__main__":
    main()
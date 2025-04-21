"""Configuration parameters for the music notation transformer."""

class Config:
    # Data parameters
    MIXED_PRECISION = True
    GRADIENT_ACCUMULATION_STEPS = 1  # Adjust as needed
    DATA_ROOT = "/import/c4dm-05/elona/agnostic-OMR/data"
    # IMAGE_SIZE = (448, 448)  # Base image size
    IMAGE_SIZE = (224, 224)
    INITIAL_SUBSET_SIZE = 10000
    
    # Patch-based processing parameters
    PATCH_BASED = True       
    PATCH_SIZE = (224, 224)  # Width, height for patches
    CONTEXT_FACTOR = 1.5     # Context factor for patches
    STAFF_INVARIANT_AUGMENTATION = True  # Apply staff-invariant augmentation
    USE_RELATIVE_POSITION = True  # Use relative position encoding
    
    # Model parameters
    VIT_MODEL = "vit_small_patch16_384"  
    HIDDEN_DIM = 384  # Small ViT hidden dimension
    NUM_ENCODER_LAYERS = 12  # Default for small ViT
    NUM_DECODER_LAYERS = 4
    NUM_HEADS = 6
    DROPOUT = 0.1
    OUTPUT_DIR = '/import/c4dm-05/elona/agnostic-OMR/logs'
    
    # Training parameters
    BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1  # Larger batch size for validation
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 30
    WARMUP_STEPS = 1000
    # GRADIENT_ACCUMULATION_STEPS = 4
    GRADIENT_ACCUMULATION_STEPS = 8 
    NUM_WORKERS = 4  # Number of data loading workers
    
    # Music notation specific
    MAX_SEQ_LENGTH = 1500
    NUM_SYMBOL_TYPES = 200  # Adjust based on your notation vocabulary
    SYMBOL_LOSS_WEIGHT = 1.0
    POSITION_LOSS_WEIGHT = 0.3  # Reduce this from default
    STAFF_POSITION_LOSS_WEIGHT = 1.0
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    VALIDATE_BATCH_SIZE = True
    # Expansion schedule
    EXPANSION_SCHEDULE = [
        (5, 2000),   # (epoch, new_size)
        (10, 4000),
        (15, 8000)
    ]
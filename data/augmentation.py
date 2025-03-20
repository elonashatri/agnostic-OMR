"""Data augmentation strategies for music notation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np
import cv2

def get_train_transforms(image_size=(448, 448)):
    """Get transformations for training."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
def staff_invariant_augmentation(image):
    """Apply augmentations that simulate staff line variations."""
    # Randomly adjust brightness/contrast
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    beta = random.uniform(-10, 10)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Randomly adjust staff line thickness
    if random.random() < 0.5:
        kernel = np.ones((1, random.randint(1, 3)), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
    
    # Randomly add noise to simulate degradation
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
    return image

def get_val_transforms(image_size=(224, 224)):
    """Get transformations for validation."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_music_specific_transforms(image_size=(224, 224)):
    """
    Get music-specific transformations.
    
    These transformations preserve the musical integrity while augmenting the data.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        # Horizontal stretching/compression (tempo variation)
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0, p=0.5),
        # Very slight rotation (page alignment variation)
        A.Rotate(limit=1, p=0.5),
        # Slight vertical stretching (staff spacing variation)
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
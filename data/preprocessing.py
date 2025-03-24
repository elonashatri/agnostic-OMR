"""Preprocessing utilities for music notation images."""

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A

def resize_with_aspect_ratio(image, target_size=(224, 224), interpolation=cv2.INTER_AREA):
    """
    Resize image while maintaining aspect ratio and filling with white background.
    
    Args:
        image: Input image (numpy array)
        target_size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image with white background
    """
    # Get original dimensions
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Create white canvas
    if len(image.shape) == 3:  # Color image
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    else:  # Grayscale image
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Calculate position to paste
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste resized image
    if len(image.shape) == 3:  # Color image
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
    else:  # Grayscale image
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def preprocess_music_score(image_path, target_size=(224, 224), to_tensor=True, normalize=True):
    """
    Preprocess a music score image for the model.
    
    Args:
        image_path: Path to the image
        target_size: Target size (width, height)
        to_tensor: Whether to convert to tensor
        normalize: Whether to normalize the image
        
    Returns:
        Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize with aspect ratio
    image = resize_with_aspect_ratio(image, target_size)
    
    # Apply transformations using Albumentations
    transform = A.Compose([
        A.Resize(height=target_size[1], width=target_size[0]),
        A.GaussianBlur(blur_limit=(1, 3), p=0.2),  # Optional blur to reduce noise
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),  # Light color jitter
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else A.NoOp(),
    ])
    
    transformed = transform(image=image)
    image = transformed['image']
    
    # Convert to tensor if requested
    if to_tensor:
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    
    return image

def binarize_score(image, threshold=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """
    Binarize a music score image to improve feature extraction.
    
    Args:
        image: Input image (numpy array)
        threshold: Threshold value
        max_value: Maximum value
        threshold_type: Thresholding type
        
    Returns:
        Binarized image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        11,  # Block size
        2    # C constant
    )
    
    return binary

def extract_staff_lines(binary_image, min_line_length=50):
    """
    Extract staff lines from a binarized music score image.
    
    Args:
        binary_image: Binarized image
        min_line_length: Minimum line length to consider
        
    Returns:
        List of staff line positions (y-coordinates)
    """
    # Apply horizontal morphological operation to connect staff line segments
    kernel = np.ones((1, 20), np.uint8)
    morph = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Apply HoughLinesP to detect lines
    lines = cv2.HoughLinesP(
        255 - morph,  # Invert image for HoughLinesP
        1,
        np.pi/180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10
    )
    
    if lines is None:
        return []
    
    # Extract horizontal lines (staff lines)
    staff_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check if line is horizontal (small y difference)
        if abs(y2 - y1) < 5:
            # Use average y position
            y_pos = (y1 + y2) // 2
            staff_lines.append(y_pos)
    
    # Remove duplicates and sort
    staff_lines = sorted(list(set(staff_lines)))
    
    return staff_lines

def normalize_positions(positions, image_size, staff_lines=None):
    """
    Normalize position coordinates relative to image size and optionally staff lines.
    
    Args:
        positions: List of [x, y, width, height] positions
        image_size: (width, height) of the image
        staff_lines: Optional list of staff line positions
        
    Returns:
        Normalized positions
    """
    width, height = image_size
    
    # Normalize by image size
    normalized = []
    for x, y, w, h in positions:
        norm_x = x / width
        norm_y = y / height
        norm_w = w / width
        norm_h = h / height
        normalized.append([norm_x, norm_y, norm_w, norm_h])
    
    # If staff lines are provided, add staff-relative positions
    if staff_lines and len(staff_lines) > 0:
        # Calculate average staff line spacing
        spacings = [staff_lines[i+1] - staff_lines[i] for i in range(len(staff_lines)-1)]
        avg_spacing = sum(spacings) / len(spacings) if spacings else 1.0
        
        # Calculate relative positions
        for i, (x, y, w, h) in enumerate(positions):
            # Find closest staff line
            closest_line = min(staff_lines, key=lambda line: abs(line - y))
            closest_idx = staff_lines.index(closest_line)
            
            # Calculate distance in staff spaces
            relative_pos = (y - closest_line) / avg_spacing
            
            # Add to normalized positions
            normalized[i].append(relative_pos)
    
    return normalized

def crop_to_staff_system(image, staff_system_bbox, padding=10):
    """
    Crop image to a specific staff system with padding.
    
    Args:
        image: Input image
        staff_system_bbox: [x, y, width, height] of the staff system
        padding: Padding around the staff system
        
    Returns:
        Cropped image
    """
    x, y, width, height = staff_system_bbox
    
    # Add padding
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(image.shape[1], x + width + padding)
    y_max = min(image.shape[0], y + height + padding)
    
    # Crop image
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped

def preprocess_notation_data(notation_data, staff_systems):
    """
    Process raw notation data into a structured format.
    
    Args:
        notation_data: Raw notation data
        staff_systems: List of staff system bounding boxes
        
    Returns:
        Processed notation data
    """
    processed_data = []
    
    # Process each staff system
    for system_idx, system_bbox in enumerate(staff_systems):
        system_items = []
        
        # Extract system ID (format: "S-{number}")
        system_id = f"S-{system_idx}"
        
        # Find items for this staff system
        for item in notation_data:
            # Skip the system item itself
            if isinstance(item, dict) and "bounding_box" in item:
                continue
            
            # Check if item belongs to this system
            if item.startswith(system_id) or f"{system_id} " in item:
                # Process item
                processed_item = process_notation_item(item)
                if processed_item:
                    system_items.append(processed_item)
        
        # Add system items to processed data
        if system_items:
            processed_data.append({
                "system_id": system_id,
                "bbox": system_bbox,
                "items": system_items
            })
    
    return processed_data

def process_notation_item(item_str):
    """
    Process a single notation item string into a structured format.
    
    Args:
        item_str: Notation item string
        
    Returns:
        Processed item as a dictionary
    """
    # Split the string by semicolons to get individual elements
    parts = item_str.split(';')
    
    # Extract system ID if present
    system_id = ""
    if parts[0].strip().startswith('S-'):
        system_part = parts[0].strip()
        system_id = system_part.split(' ')[0]
        parts = parts[1:]  # Remove system part
    
    processed_items = []
    
    # Process each part
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Extract symbol type and position info
        if '-' in part and '{' in part:
            symbol_info = {}
            
            # Extract symbol type
            symbol_type = part.split('-')[0].strip()
            symbol_info['type'] = symbol_type
            
            # Extract staff position (L1, S2, etc.) if present
            if len(part.split('-')) > 1 and part.split('-')[1].startswith(('L', 'S')):
                staff_pos = part.split('-')[1].split('{')[0].strip()
                symbol_info['staff_position'] = staff_pos
            
            # Extract position coordinates
            if '{' in part and '}' in part:
                pos_str = part[part.find('{'):part.find('}')+1]
                try:
                    # Parse the position string
                    pos_dict = eval(pos_str.replace("'", '"'))
                    symbol_info['position'] = [
                        pos_dict.get('l', 0),
                        pos_dict.get('t', 0),
                        pos_dict.get('w', 0),
                        pos_dict.get('h', 0)
                    ]
                except:
                    # If parsing fails, use default values
                    symbol_info['position'] = [0, 0, 0, 0]
            
            processed_items.append(symbol_info)
    
    return processed_items

def augment_music_score(image, augmentation_strength='medium'):
    """
    Apply data augmentation to a music score image.
    
    Args:
        image: Input image
        augmentation_strength: Strength of augmentation ('mild', 'medium', or 'strong')
        
    Returns:
        Augmented image
    """
    # Define augmentation parameters based on strength
    if augmentation_strength == 'mild':
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.Rotate(limit=1, p=0.5),  # Very slight rotation
        ])
    elif augmentation_strength == 'medium':
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
            A.Rotate(limit=2, p=0.7),
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),  # Staff spacing variation
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
        ])
    elif augmentation_strength == 'strong':
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
            A.Rotate(limit=3, p=0.8),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.5),
            A.Blur(blur_limit=3, p=0.3),
        ])
    else:
        # Default to medium if invalid strength is provided
        return augment_music_score(image, 'medium')
    
    # Apply augmentation
    augmented = transform(image=image)
    
    return augmented['image']

def save_sample_patches(patches, symbol_types, base_dir="sample_patches", num_samples=5):
    """Save sample patches for visualization."""
    import os
    import cv2
    import random
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Select random samples or take first few if less than requested
    num_to_save = min(num_samples, len(patches))
    if num_to_save == 0:
        return
    
    indices = random.sample(range(len(patches)), num_to_save)
    
    # Save each sample
    for i, idx in enumerate(indices):
        patch = patches[idx]
        symbol, staff_pos = symbol_types[idx]
        
        # Create a unique filename
        filename = f"{base_dir}/sample_{i}_{symbol}_{staff_pos}.png"
        
        # Convert to BGR for OpenCV
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            # RGB to BGR for color images
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        else:
            patch_bgr = patch
            
        # Save the image
        cv2.imwrite(filename, patch_bgr)
        print(f"Saved sample patch: {filename}")
        
def extract_notation_patches(image, notation_data, patch_size=(448, 448), context_factor=1.5):
    """Extract patches around notation elements with context."""
    patches = []
    positions = []
    symbol_types = []
    
    print(f"Processing notation data with {len(notation_data)} items")
    
    for item_idx, item in enumerate(notation_data):
        # For debugging, let's print some sample items
        if item_idx < 3:
            print(f"Sample item {item_idx}: {item[:100]}...")
        
        # Check if this is a system entry
        is_system = False
        if isinstance(item, str) and 'bounding_box' in item and item.startswith('S-'):
            is_system = True
            if item_idx < 3:
                print(f"Found system entry: {item[:100]}...")
            
            # Extract the content part after the bounding box
            if ';' in item:
                # The part after the first semicolon contains the musical elements
                content_part = item[item.find(';')+1:]
                
                # Process individual elements in this system
                for element in content_part.split(';'):
                    element = element.strip()
                    if not element or not ('{' in element and '}' in element):
                        continue
                    
                    try:
                        # Extract position info
                        pos_str = element[element.find('{'):element.find('}')+1]
                        pos_dict = eval(pos_str.replace("'", '"'))
                        x, y = float(pos_dict.get('l', 0)), float(pos_dict.get('t', 0))
                        w, h = float(pos_dict.get('w', 0)), float(pos_dict.get('h', 0))
                        
                        # Extract symbol type and staff position
                        parts = element.strip().split('-')
                        if len(parts) > 0:
                            symbol_type = parts[0].strip()
                            
                            # Get staff position if available
                            staff_pos = None
                            if len(parts) > 1:
                                for part in parts[1:]:
                                    if part.startswith('L') or part.startswith('S'):
                                        staff_pos = part.split('{')[0].strip()
                                        break
                            
                            # Skip elements with very small size
                            if w < 5 or h < 5:
                                continue
                                
                            # Ensure coordinates are within image bounds
                            x = max(0, min(x, image.shape[1]-1))
                            y = max(0, min(y, image.shape[0]-1))
                            w = max(1, min(w, image.shape[1]-x))
                            h = max(1, min(h, image.shape[0]-y))
                            
                            # Extract patch with context
                            context_w = max(w * context_factor, 32)
                            context_h = max(h * context_factor, 32)
                            
                            patch_x = max(0, int(x - (context_w - w)/2))
                            patch_y = max(0, int(y - (context_h - h)/2))
                            patch_w = min(image.shape[1] - patch_x, int(context_w))
                            patch_h = min(image.shape[0] - patch_y, int(context_h))
                            
                            # Extract and resize patch
                            try:
                                patch = image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                                patch = cv2.resize(patch, patch_size)
                                
                                patches.append(patch)
                                positions.append([x, y, w, h])
                                symbol_types.append((symbol_type, staff_pos))
                            except Exception as e:
                                print(f"Error creating patch: {e}, Coords: {patch_x}:{patch_y+patch_h}, {patch_x}:{patch_x+patch_w}")
                            
                    except Exception as e:
                        print(f"Error processing element: {element[:50]}..., Error: {e}")
        
        # If not a system entry, process as before (for backward compatibility)
        if not is_system:
            # Process individual elements
            element_count = 0
            for element in item.split(';'):
                # Existing element processing code...
                # ...
                element_count += 1
            
            # print(f"Item {item_idx}: Processed {element_count} elements")
    
    # print(f"Extracted {len(patches)} patches with {len(set([s[0] for s in symbol_types]))} unique symbol types")
    if symbol_types:
        print(f"Sample symbols: {[s[0] for s in symbol_types[:5]]}")
    
    # After extracting all patches
    if len(patches) > 0:
        save_sample_patches(patches, symbol_types, base_dir="sample_patches", num_samples=1)
    return patches, positions, symbol_types

def create_relative_position_encoding(notation_data):
    """Create relative position encoding for symbols."""
    symbol_positions = []
    
    # First, collect all symbol positions
    for item in notation_data:
        if 'bounding_box' in str(item):
            continue
            
        for element in item.split(';'):
            element = element.strip()
            if not element or not ('{' in element and '}' in element):
                continue
                
            if '{' in element and '}' in element:
                pos_str = element[element.find('{'):element.find('}')+1]
                try:
                    pos_dict = eval(pos_str.replace("'", '"'))
                    x = float(pos_dict.get('l', 0))
                    y = float(pos_dict.get('t', 0))
                    symbol_positions.append((x, y))
                except:
                    pass
    
    # Cluster vertical positions (likely staff lines)
    y_positions = [pos[1] for pos in symbol_positions]
    
    # Simple clustering algorithm
    def cluster_y_positions(y_positions, threshold=10):
        """Cluster y positions with a threshold."""
        if not y_positions:
            return []
            
        # Sort positions
        sorted_positions = sorted(y_positions)
        
        # Initialize clusters
        clusters = [[sorted_positions[0]]]
        
        # Assign positions to clusters
        for pos in sorted_positions[1:]:
            if pos - clusters[-1][-1] < threshold:
                # Add to current cluster
                clusters[-1].append(pos)
            else:
                # Start new cluster
                clusters.append([pos])
        
        # Calculate cluster centers
        cluster_centers = [sum(cluster) / len(cluster) for cluster in clusters]
        
        return cluster_centers
    
    clustered_positions = cluster_y_positions(y_positions)
    
    # Calculate relative positions
    def calculate_relative_position(y, cluster_centers):
        """Calculate relative position to nearest clusters."""
        if not cluster_centers:
            return 0.0
            
        # Find nearest cluster
        distances = [abs(y - center) for center in cluster_centers]
        nearest_idx = distances.index(min(distances))
        
        # Find distance to nearest and next nearest cluster
        if nearest_idx > 0 and nearest_idx < len(cluster_centers) - 1:
            above_dist = y - cluster_centers[nearest_idx - 1]
            below_dist = cluster_centers[nearest_idx + 1] - y
            
            # Normalize by staff space
            staff_space = (cluster_centers[nearest_idx + 1] - cluster_centers[nearest_idx - 1]) / 2
            if staff_space > 0:
                relative_pos = (y - cluster_centers[nearest_idx]) / staff_space
                return relative_pos
        
        # Fallback: calculate simple relative position
        nearest_center = cluster_centers[nearest_idx]
        return (y - nearest_center) / 10.0  # Arbitrary scaling
    
    # Assign relative positions to all symbols
    relative_positions = []
    for pos in symbol_positions:
        relative_y = calculate_relative_position(pos[1], clustered_positions)
        relative_positions.append(relative_y)
        
    return relative_positions


def parse_staff_position(position_str):
    """
    Parse staff position string (L1, S2, etc.) to numeric value.
    
    Args:
        position_str: Staff position string
        
    Returns:
        Numeric value for the staff position
    """
    if not position_str:
        return 0
    
    try:
        # Extract the type (Line or Space) and number
        type_char = position_str[0]
        number = int(position_str[1:])
        
        if type_char == 'L':
            # Line positions (1-5 from bottom to top)
            position = number - 3  # Map to -2, -1, 0, 1, 2
        elif type_char == 'S':
            # Space positions (1-4 from bottom to top)
            position = number - 2.5  # Map to -1.5, -0.5, 0.5, 1.5
        else:
            position = 0
            
        return position
    except:
        return 0
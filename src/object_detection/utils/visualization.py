"""Visualization utilities for object detection results."""

from typing import Optional, List, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from object_detection.models.base import DetectionResult


def visualize_detections(
    image: Union[Image.Image, np.ndarray],
    result: DetectionResult,
    show_labels: bool = True,
    show_scores: bool = True,
    color: str = "red",
    thickness: int = 2,
) -> Image.Image:
    """
    Visualize detection results on an image.
    
    Args:
        image: Input image
        result: Detection result
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores
        color: Color for bounding boxes
        thickness: Line thickness for boxes
        
    Returns:
        Image with visualized detections
    """
    # Convert to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create a copy
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for i in range(len(result.boxes)):
        box = result.boxes[i]
        score = result.scores[i]
        label_idx = result.labels[i]
        
        # Draw bounding box
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline=color,
            width=thickness,
        )
        
        # Prepare label text
        label_text = ""
        if show_labels and result.class_names:
            if label_idx < len(result.class_names):
                label_text = result.class_names[label_idx]
        
        if show_scores:
            score_text = f"{score:.2f}"
            label_text = f"{label_text} {score_text}" if label_text else score_text
        
        # Draw label
        if label_text:
            # Get text size
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle for text
            draw.rectangle(
                [(box[0], box[1] - text_height - 4),
                 (box[0] + text_width + 4, box[1])],
                fill=color,
            )
            
            # Draw text
            draw.text(
                (box[0] + 2, box[1] - text_height - 2),
                label_text,
                fill="white",
                font=font,
            )
    
    return img_copy


def create_detection_grid(
    images: List[Union[Image.Image, np.ndarray]],
    results: List[DetectionResult],
    grid_cols: int = 3,
    **viz_kwargs
) -> Image.Image:
    """
    Create a grid of visualized detections.
    
    Args:
        images: List of input images
        results: List of detection results
        grid_cols: Number of columns in the grid
        **viz_kwargs: Additional arguments for visualize_detections
        
    Returns:
        Grid image
    """
    # Visualize all images
    viz_images = [
        visualize_detections(img, result, **viz_kwargs)
        for img, result in zip(images, results)
    ]
    
    # Calculate grid dimensions
    n_images = len(viz_images)
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Get max dimensions
    max_width = max(img.width for img in viz_images)
    max_height = max(img.height for img in viz_images)
    
    # Create grid
    grid_width = grid_cols * max_width
    grid_height = grid_rows * max_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(viz_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * max_width
        y = row * max_height
        grid.paste(img, (x, y))
    
    return grid

"""
CST435: Parallel and Cloud Computing - Assignment 2
Image Processing Filters Module

This module implements the 5 required image filters:
1. Grayscale Conversion - Using luminance formula
2. Gaussian Blur - 3x3 kernel for smoothing
3. Sobel Edge Detection - Detects edges using Sobel operator
4. Image Sharpening - Enhances edges and details
5. Brightness Adjustment - Increases/decreases brightness

Parallelization Strategy:
- Each filter operates on individual pixels or small neighborhoods
- Filters are applied sequentially to each image (filter pipeline)
- Different IMAGES are processed in parallel (data parallelism)
- No shared state between image processing tasks = no synchronization needed
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def grayscale_filter(image):
    """
    Filter 1: Grayscale Conversion using Luminance Formula
    
    Converts RGB to grayscale using weighted sum:
    Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
    
    Args:
        image: PIL Image in RGB mode
    Returns:
        PIL Image in grayscale (L mode)
    """
    return ImageOps.grayscale(image)


def gaussian_blur_filter(image):
    """
    Filter 2: Gaussian Blur (3x3 kernel)
    
    Applies smoothing to reduce noise and detail.
    Uses a 3x3 Gaussian kernel approximation.
    
    Args:
        image: PIL Image
    Returns:
        PIL Image with Gaussian blur applied
    """
    # Pillow's GaussianBlur with radius=1 approximates a 3x3 kernel
    return image.filter(ImageFilter.GaussianBlur(radius=1))


def sobel_edge_detection_filter(image):
    """
    Filter 3: Sobel Edge Detection
    
    Detects edges using Sobel operator:
    - Computes horizontal gradient (Gx)
    - Computes vertical gradient (Gy)
    - Magnitude = sqrt(Gx² + Gy²)
    
    Args:
        image: PIL Image (grayscale works best)
    Returns:
        PIL Image with edges highlighted
    """
    # Convert PIL Image to OpenCV format (numpy array)
    opencv_image = np.array(image)

    # Apply Sobel filter in X and Y directions
    sobelx = cv2.Sobel(opencv_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(opencv_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert back to PIL Image
    return Image.fromarray(magnitude)


def sharpen_filter(image):
    """
    Filter 4: Image Sharpening
    
    Enhances edges and fine details using a sharpening kernel.
    Makes the image appear crisper.
    
    Args:
        image: PIL Image
    Returns:
        PIL Image with sharpening applied
    """
    return image.filter(ImageFilter.SHARPEN)


def brightness_adjustment_filter(image, factor=1.2):
    """
    Filter 5: Brightness Adjustment
    
    Adjusts image brightness by a factor.
    factor > 1.0 increases brightness
    factor < 1.0 decreases brightness
    
    Args:
        image: PIL Image
        factor: Brightness multiplier (default 1.2 = 20% brighter)
    Returns:
        PIL Image with adjusted brightness
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_filters(image):
    """
    Applies the complete filter pipeline to an image.
    
    Pipeline Order:
    1. Grayscale → 2. Gaussian Blur → 3. Sobel Edge → 4. Sharpen → 5. Brightness
    
    This function is called by each parallel worker (process or thread).
    Since each image is processed independently, no synchronization is needed.
    
    Args:
        image: PIL Image (any mode)
    Returns:
        PIL Image after all 5 filters applied
    """
    # Ensure the image is in RGB mode for consistent processing
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the 5 specified filters in sequence
    image = grayscale_filter(image)           # Filter 1
    image = gaussian_blur_filter(image)       # Filter 2
    image = sobel_edge_detection_filter(image) # Filter 3
    image = sharpen_filter(image)             # Filter 4
    image = brightness_adjustment_filter(image) # Filter 5
    
    return image

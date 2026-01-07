import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

def grayscale_filter(image):
    """Applies a grayscale filter to the image (Luminance)."""
    return ImageOps.grayscale(image)

def gaussian_blur_filter(image):
    """Applies a Gaussian Blur filter (3x3) to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=1)) # Pillow's GaussianBlur with radius 1 approximates a 3x3 kernel

def sobel_edge_detection_filter(image):
    """Applies Sobel Edge Detection to the image."""
    # Convert PIL Image to OpenCV format
    # Convert PIL Image to OpenCV format (already grayscale from previous filter)
    opencv_image = np.array(image)

    # Apply Sobel filter
    sobelx = cv2.Sobel(opencv_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(opencv_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert back to PIL Image
    return Image.fromarray(magnitude)

def sharpen_filter(image):
    """Applies a sharpen filter to the image."""
    return image.filter(ImageFilter.SHARPEN)

def brightness_adjustment_filter(image, factor=1.2):
    """Adjusts the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_filters(image):
    """Applies a sequence of predefined filters to an image."""
    # Ensure the image is in RGB mode for consistent processing
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the 5 specified filters
    image = grayscale_filter(image)
    image = gaussian_blur_filter(image)
    image = sobel_edge_detection_filter(image)
    image = sharpen_filter(image)
    image = brightness_adjustment_filter(image)
    return image

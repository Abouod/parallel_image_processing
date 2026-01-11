import os
import random
import shutil
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_food101_subset(path, num_images=100):
    """
    Downloads a subset of images from the Food-101 dataset.
    This function will download images from a predefined list of categories
    to simulate the Food-101 dataset without downloading the entire large dataset.
    """
    if os.path.exists(path):
        print(f"Dataset already exists at {path}. Skipping download.")
        return

    os.makedirs(path, exist_ok=True)
    base_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos/"
    
    # Using a small, publicly available dataset for demonstration purposes.
    # Food-101 is very large, so we'll use placeholder images from picsum.photos
    # to simulate the image processing pipeline.
    image_urls = []
    for i in range(num_images):
        # Generate random image URLs from placehold.co
        # Using different dimensions to ensure variety
        width = random.randint(400, 800)
        height = random.randint(300, 600)
        image_urls.append(f"https://placehold.co/{width}x{height}/png?text=Image+{i}")

    # Shuffle and take a subset if more URLs were generated than needed
    random.shuffle(image_urls)
    image_urls = image_urls[:num_images]

    print(f"Downloading {len(image_urls)} images...")
    for i, url in enumerate(tqdm(image_urls)):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            # Convert to RGB if not already, as JPEG does not support RGBA
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Save with a generic name, or parse filename from URL if desired
            img.save(os.path.join(path, f"image_{i:04d}.jpg"))
        except Exception as e:
            print(f"Could not download or save image from {url}: {e}")

def load_images(path, num_images=None):
    """Loads images from a specified directory.
    
    Args:
        path: Directory path containing images
        num_images: Maximum number of images to load (None = load all)
    """
    images_data = []
    if not os.path.exists(path):
        print(f"Dataset path {path} does not exist.")
        return images_data

    filenames = sorted([f for f in os.listdir(path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    
    # Limit to num_images if specified
    if num_images is not None:
        filenames = filenames[:num_images]
    
    for filename in filenames:
        filepath = os.path.join(path, filename)
        try:
            img = Image.open(filepath)
            img.load()  # Load image data into memory
            images_data.append((img, filename))
        except Exception as e:
            print(f"Could not load image {filepath}: {e}")
    return images_data

def save_image(image, filepath):
    """Saves a PIL Image object to a specified filepath."""
    try:
        image.save(filepath)
    except Exception as e:
        print(f"Could not save image to {filepath}: {e}")

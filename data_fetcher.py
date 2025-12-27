"""
Satellite Imagery Data Fetcher

This script downloads satellite images for properties using their latitude and longitude coordinates.
Uses Google Maps tile server to fetch and stitch satellite imagery centered on each property.

Author: Data Science Project
Date: 2024
"""

import os
import math
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO


# --- CONFIGURATION ---
OUTPUT_FOLDER = "property_images"  # Change this to your desired output folder
ZOOM_LEVEL = 19  # Higher zoom = more detail (19 is recommended for property-level detail)
IMAGE_SIZE = 512  # Standard ML input size
MAX_WORKERS = 12  # Number of parallel downloads
CSV_PATH = "data/train(1)(train(1)).csv"  # Path to your training data


def get_tile_coords(lat, lon, zoom):
    """
    Calculates the exact pixel location on the global map.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (higher = more detail)
    
    Returns:
        tuple: (x, y) pixel coordinates on the global map
    """
    scale = 1 << zoom
    siny = math.sin(lat * math.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y


def fetch_centered_house(row):
    """
    Fetches and saves a centered satellite image for a property.
    
    Args:
        row: Dictionary containing property information with 'id', 'lat', 'long' keys
    """
    prop_id = row['id']
    save_path = os.path.join(OUTPUT_FOLDER, f"{prop_id}.png")
    
    # Skip if image already exists
    if os.path.exists(save_path):
        return

    try:
        # 1. Get precise global pixel coordinates
        world_x, world_y = get_tile_coords(row['lat'], row['long'], ZOOM_LEVEL)
        
        # 2. Determine the main tile and pixel offset
        tile_x, tile_y = int(world_x), int(world_y)
        offset_x = int((world_x - tile_x) * 256)
        offset_y = int((world_y - tile_y) * 256)

        # 3. Create a canvas and stitch 2x2 tiles to ensure the house is centered
        # This prevents the house from being 'cut off' by a tile edge
        canvas = Image.new('RGB', (512, 512))
        
        for i in range(2):
            for j in range(2):
                # Using Google's Public Tile Server (No API Key needed)
                url = f"https://mt1.google.com/vt/lyrs=s&x={tile_x + i}&y={tile_y + j}&z={ZOOM_LEVEL}"
                response = requests.get(url, timeout=5)
                tile = Image.open(BytesIO(response.content))
                canvas.paste(tile, (i * 256, j * 256))

        # 4. Crop the canvas so the house is exactly in the middle
        # We take a crop centered on our calculated offset
        left = offset_x
        top = offset_y
        final_img = canvas.crop((left, top, left + 256, top + 256))
        
        # Resize to standard ML input size
        final_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(save_path)
    except Exception as e:
        # Silently fail for individual properties to continue processing
        pass


def main():
    """
    Main function to download satellite images for all properties in the dataset.
    """
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load property data
    print(f"Loading data from {CSV_PATH}...")
    df_properties = pd.read_csv(CSV_PATH)
    print(f"Found {len(df_properties)} properties")
    
    # Check for required columns
    required_cols = ['id', 'lat', 'long']
    missing_cols = [col for col in required_cols if col not in df_properties.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\nStitching centered images at Zoom {ZOOM_LEVEL}...")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print(f"Max workers: {MAX_WORKERS}\n")
    
    # Download images in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(tqdm(
            pool.map(fetch_centered_house, df_properties.to_dict('records')), 
            total=len(df_properties),
            desc="Downloading images"
        ))
    
    # Count successfully downloaded images
    downloaded = sum(1 for _ in os.listdir(OUTPUT_FOLDER) if _.endswith('.png'))
    print(f"\n✓ Download complete!")
    print(f"✓ Successfully downloaded {downloaded} images")
    print(f"✓ Images saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()


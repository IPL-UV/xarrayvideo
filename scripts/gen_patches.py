import os
from PIL import Image
import numpy as np
from enum import Enum

class CropPosition(Enum):
    CENTER = "center"
    TOPLEFT = "topleft"
    TOPRIGHT = "topright"
    BOTTOMLEFT = "bottomleft"
    BOTTOMRIGHT = "bottomright"

def process_images(original_path, crop_size=40, reduction_factor=4, crop_position=CropPosition.CENTER):
    """
    Process original and compressed images with cropping and full image reduction.
    
    Args:
        original_path (str): Path to the original image
        crop_size (int): Size of the crop (square)
        reduction_factor (int): Factor by which to reduce the full image
        crop_position (CropPosition): Position to take the crop from
    """
    qualities = ['Best', 'Very high', 'High', 'Medium', 'Low', 'Very low']
    base_dir = os.path.dirname(original_path)
    
    def get_crop_coordinates(width, height, position):
        """Calculate crop coordinates based on position"""
        if position == CropPosition.CENTER:
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
        elif position == CropPosition.TOPLEFT:
            left = 0
            top = 0
        elif position == CropPosition.TOPRIGHT:
            left = width - crop_size
            top = 0
        elif position == CropPosition.BOTTOMLEFT:
            left = 0
            top = height - crop_size
        elif position == CropPosition.BOTTOMRIGHT:
            left = width - crop_size
            top = height - crop_size
        
        right = left + crop_size
        bottom = top + crop_size
        return left, top, right, bottom

    def create_crop(image_path, output_suffix='_crop'):
        """Create crop of an image and save it"""
        img = Image.open(image_path)
        width, height = img.size
        
        # Get crop coordinates based on position
        left, top, right, bottom = get_crop_coordinates(width, height, crop_position)
        
        # Crop and save
        cropped = img.crop((left, top, right, bottom))
        output_path = image_path.replace('.png', f'{output_suffix}_{crop_position.value}.png')\
            .replace('/samples', '/cropped')
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cropped.save(output_path)
        print(f"Saved cropped image: {output_path}")
        return output_path

    def reduce_full_image(image_path, output_suffix='_non_cropped'):
        """Reduce the size of the full image"""
        img = Image.open(image_path)
        width, height = img.size
        new_size = (width // reduction_factor, height // reduction_factor)
        reduced = img.resize(new_size, Image.Resampling.LANCZOS)
        output_path = image_path.replace('.png', f'{output_suffix}.png')\
            .replace('/samples', '/cropped')
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        reduced.save(output_path)
        print(f"Saved reduced image: {output_path}")
        return output_path

    # Process original image
    create_crop(original_path)
    reduce_full_image(original_path)

    # Process compressed versions
    for quality in qualities:
        compressed_path = original_path.replace('Best', quality).replace('orig', 'comp')
        if os.path.exists(compressed_path):
            create_crop(compressed_path)
            #reduce_full_image(compressed_path)
        else:
            print(f"Warning: Compressed image for quality {quality} not found: {compressed_path}")

if __name__ == "__main__":
    original_image= "results/samples/dynamicearthnet_img_libx265_12_8077_5007_13_bands_Best_60_orig.png"   
    process_images(
        original_path=original_image,
        crop_size=120,
        reduction_factor=4,
        crop_position=CropPosition.TOPLEFT
    )
    
    # original_image= "results/samples/deepextremes_img_libx265_12_95.29_29.09_rgb_Best_120_orig.png"
    original_image= "results/samples/deepextremes_img_libx265_12_10.38_50.15_rgb_Best_120_orig.png"
    process_images(
        original_path=original_image,
        crop_size=120,
        reduction_factor=1,
        crop_position=CropPosition.CENTER
    )
    
    original_image= "results/samples/custom_img_libx265_12_..cubos_juliocubo1_pickle_all_Best_1_orig.png"
    process_images(
        original_path=original_image,
        crop_size=120,
        reduction_factor=4,
        crop_position=CropPosition.CENTER
    )
    
    # original_image= "results/samples/era5_img_libx265_12_ERA5_0_relative_humidity_Best_40_orig.png"
    original_image= "results/samples/era5_img_libx265_12_ERA5_0_wind_speed_Best_20_orig.png"
    process_images(
        original_path=original_image,
        crop_size=120,
        reduction_factor=4,
        crop_position=CropPosition.CENTER
    )
import os
import tensorflow as tf

def load_image(image_path):
    """Load an image, decode it, and resize to 256x256"""
    img = tf.io.read_file(image_path)               # Read image
    img = tf.image.decode_jpeg(img, channels=3)       # Decode as JPG (assuming JPG images)
    img = tf.image.resize(img, (256, 256))           # Resize to target size
    img = img / 255.0                               # Normalize to range [0, 1]
    return img

def load_mask(mask_path):
    """Load a mask, decode it, and resize to 256x256"""
    mask = tf.io.read_file(mask_path)               # Read mask image
    mask = tf.image.decode_png(mask, channels=1)     # Decode as PNG (assuming masks are single-channel)
    mask = tf.image.resize(mask, (256, 256))         # Resize to target size
    mask = mask / 255.0                             # Normalize to range [0, 1]
    return mask


# Set the directory paths
image_folder = 'normalized_images'
mask_folder = 'normalized_masks'

# Get image and mask file paths
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
mask_paths = [os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)]

'''
for image_path, mask_path in zip(image_paths, mask_paths):
    image = load_image(image_path)
    mask = load_mask(mask_path)
    print(f"Loaded image: {image.shape}, mask: {mask.shape}")
'''

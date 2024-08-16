import os
from PIL import Image
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

# Directory containing the images
path_file = r'C:\Users\DELL\Desktop\my new project\tiger_data\train\not_tiger'  # Update with your actual path

# Directory where you want to save the resized images (e.g., Desktop)
save_directory = r'C:\Users\DELL\Desktop\my new project\tiger_data\train\not_tiger_1'  # Update with your actual path

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Process and save images
for filename in os.listdir(path_file):
    # Read the image using skimage
    img = imread(os.path.join(path_file, filename))

    # Resize the image
    img_resized = resize(img, (128, 128))

    # Convert the image to uint8
    img_uint8 = (img_resized * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    pil_img = Image.fromarray(img_uint8)

    # Save the image using PIL
    save_path = os.path.join(save_directory, filename)
    pil_img.save(save_path)

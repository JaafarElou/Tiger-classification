import os
from PIL import Image
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

path_file = r'C:\Users\DELL\Desktop\my new project\tiger_data\train\not_tiger'  

save_directory = r'C:\Users\DELL\Desktop\my new project\tiger_data\train\not_tiger_1'

os.makedirs(save_directory, exist_ok=True)

for filename in os.listdir(path_file):

    img = imread(os.path.join(path_file, filename))

    img_resized = resize(img, (128, 128))

    img_uint8 = (img_resized * 255).astype(np.uint8)

    pil_img = Image.fromarray(img_uint8)

    save_path = os.path.join(save_directory, filename)
    
    pil_img.save(save_path)

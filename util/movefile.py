import os
import shutil
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

jpg_dir = r"F:\HSI\seg\datasets\imgs\train"
png_dir = r"F:\HSI\seg\datasets\labels\train"

#save png
save_png = r"F:\HSI\seg\train1"
if not os.path.exists(save_png):
    os.mkdir(save_png)
jpgs = os.listdir(jpg_dir)

for jpg in jpgs:
    jpg_name = jpg[:-4]
    png_path = os.path.join(png_dir, jpg_name) + '.png'
    # img = Image.open(png_path).convert('P')
    # img.save(os.path.join(save_png, jpg_name+'.png'))
    shutil.copy(png_path, os.path.join(save_png, jpg_name+'.png'))
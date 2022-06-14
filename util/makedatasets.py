import os
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import imgviz

"""
对图片进行剪切一半
"""
img_dir = r"F:\HSI\seg\JPEGImages"
mask_dir = r"F:\HSI\seg\SegmentationClass"

img_dir = os.path.join(img_dir)
img_list = os.listdir(img_dir)

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    mask_path = os.path.join(mask_dir, img_name[:-4]+".png")
    mask = Image.open(mask_path)
    # import tifffile as tiff
    # img = tiff.imread(img_path)
    # img = Image.fromarray(img).convert("RGB")

    img = np.asarray(img)
    mask = np.asarray(mask)
    sp = img.shape[1]//2
    img1 = img[:,:sp]
    img2 = img[:, sp:]
    mask1 = mask[:, :sp]
    mask2 = mask[:, sp:]

    # save img
    train_path = "F:/HSI/seg/data/train/" + img_name
    test_path = "F:/HSI/seg/data/test/" + img_name
    img1 = Image.fromarray(img1)
    img1.save(train_path)
    img2 = Image.fromarray(img2)
    img2.save(test_path)

    # save mask
    train_path = "F:/HSI/seg/label/train/" + img_name[:-4]+".png"
    test_path = "F:/HSI/seg/label/test/" + img_name[:-4]+".png"

    lbl_pil = Image.fromarray(mask1.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(train_path)

    lbl_pil = Image.fromarray(mask2.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(test_path)





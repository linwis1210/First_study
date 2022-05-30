import os
import shutil
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import imgviz

"""
对labelme生成的标签图片进行切割
"""
label_dir = r"F:\HSI\seg\SegmentationClass"
label_names = os.listdir(label_dir)

num = 1
for label_name in label_names:
    img = Image.open(os.path.join(label_dir, label_name))
    [src_h, src_w] = img.size
    [src_h, src_w] = src_h - src_h % 1000, src_w - src_w % 1000
    img = img.resize((src_h, src_w), Image.BILINEAR)

    img = np.asarray(img)
    cut_size = [1000, 1000]
    [row, col] = img.shape

    h = row // cut_size[0]
    w = col // cut_size[1]

    img_path = "F:/HSI/seg/png2/" + str(num) + "_"
    num = num + 1
    img_name = 0
    for i in range(h):
        for j in range(w):
            ROI = img[i*cut_size[0]:(i+1)*cut_size[0], j*cut_size[1] : (j+1)*cut_size[1]]
            lbl_pil = Image.fromarray(ROI.astype(np.uint8), mode="P")
            colormap = imgviz.label_colormap()
            lbl_pil.putpalette(colormap.flatten())
            lbl_pil.save(img_path + str(img_name)+".png")

            img_name = img_name + 1







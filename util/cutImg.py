import os
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

"""
对图片进行切割
"""
subdir = "test"
img_dir = r"F:\HSI\seg\data"

img_dir = os.path.join(img_dir, subdir)
img_list = os.listdir(img_dir)

num = 1
for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    # img = Image.open(img_path).convert("RGB")

    import tifffile as tiff
    img = tiff.imread(img_path)
    img = Image.fromarray(img).convert("RGB")

    [src_h, src_w] = img.size
    [src_h, src_w] = src_h - src_h % 1000, src_w - src_w % 1000
    print(src_w, src_h)
    img = img.resize((src_h, src_w), Image.BILINEAR)

    img = np.asarray(img)
    cut_size = [1000, 1000]
    [row, col, _] = img.shape

    h = row // cut_size[0]
    w = col // cut_size[1]

    img_path = "F:/HSI/seg/datasets/imgs/" + subdir
    log_name = os.path.join(img_path, f"info.txt")
    os.makedirs(img_path, exist_ok=True)
    img_path = img_path + "/" + str(num) + "_"

    # 写入resize、cut size
    with open(log_name, 'w') as f:
        f.write(f"resize size:{src_w, src_h}\n" f"cut size :{cut_size}")
    f.close()
    num = num+1
    img_name = 0
    for i in tqdm(range(h)):
        for j in range(w):
            ROI = img[i*cut_size[0]:(i+1)*cut_size[0], j*cut_size[1] : (j+1)*cut_size[1], :]
            ROI_img = Image.fromarray(ROI)
            ROI_img.save(img_path + str(img_name)+".jpg")
            img_name = img_name + 1







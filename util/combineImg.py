import os
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

"""
对图片进行合并
"""
subdir = "15"
img_dir = "F:/HSI/seg/result/" + subdir
img_list = []
for img_l in os.listdir(img_dir):
    if img_l.endswith(".png"):
        img_list.append(img_l)
# 对文件进行排序  xx_yy : XX代表大图的名字， yy代表当前图片切割后第几张小图
img_list.sort(key=lambda x: int(x.split("_")[1][:-4]))
print(img_list)

src_img_path = "F:/HSI/seg/combine/" + subdir + ".jpg"

with open(img_dir+"/info.txt", "r") as f:
    line = f.readline()
    line = line.split(":")[1].replace(" ", "")
    line = line.replace("\n", "")
    src_h = int(line.split(",")[0][1:])
    src_w = int(line.split(",")[1][:-1])
f.close()

cut_size = [1000, 1000]  # 之前切割成小图的大小
h = src_h // cut_size[0]
w = src_w // cut_size[1]

src_img = []
num = 0
for i in tqdm(range(h)):
    img_w = []
    for j in range(w):
        imag = Image.open(os.path.join(img_dir, img_list[num]))
        num = num + 1
        img = np.asarray(imag)
        img_w.append(img)
    img_w = np.concatenate(img_w, axis=1)
    src_img.append(img_w)
src_img = np.concatenate(src_img, axis=0)
src_img = Image.fromarray(src_img)
src_img.show()
src_img.save(src_img_path)











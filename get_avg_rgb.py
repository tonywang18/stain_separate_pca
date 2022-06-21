
import os


import numpy as np

from cut_patch_mask import slide_sample_list
from utils.openslide_utils import Slide
from utils.tissue_utils import get_tissue


import matplotlib.pyplot as plt
from skimage import io
import cv2

img_path = "/media/totem_disk/totem/Lung_HE_20220401/Set2_transform/IHC/13935f 35010.svs"

#讀取幻燈片
slide = Slide(img_path)
#掩碼級數爲2
mask_level = 2
svs_im = np.array(slide.get_thumb(mask_level))[:,:,:3]
#進行兩級下採樣
slide.get_level_downsample(mask_level)
#**2爲取平方
contour_area_threshold = 10000/(slide.get_level_downsample(mask_level)/slide.get_level_downsample(2))**2
svs_regin_mask, _ = get_tissue(svs_im, contour_area_threshold)
print(svs_regin_mask.shape)
plt.imshow(svs_regin_mask)
plt.show()
#把有组织的坐标保存到data_lib[]中
data_lib,patch_C,patch_R = slide_sample_list(slide,svs_regin_mask)
# print(patch_C)
# print(len(data_lib))
print("------------------------開始進行分割----------------------------")
sum_r_max,sum_g_max,sum_b_max=0,0,0
rgb_avg=[]
avg_r,avg_g,avg_b=0,0,0
for i in range(len(data_lib)):

    sum_r,sum_g,sum_b=0,0,0

    [cord, patch_index] = data_lib[i]
    img = slide.read_region(cord, 0, (256, 256)).convert("RGB")
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            pix=img.getpixel((x,y))
            sum_r+=pix[0]
            sum_g+=pix[1]
            sum_b+=pix[2]
    avg_r,avg_g,avg_b=sum_r/(x*y),sum_g/(x*y),sum_b/(x*y)

    rgb_avg.append(avg_r)
    rgb_avg.append(avg_g)
    rgb_avg.append(avg_b)
    print(rgb_avg)
    break



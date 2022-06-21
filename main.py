
import os


import numpy as np

from cut_patch_mask import slide_sample_list
from utils.openslide_utils import Slide
from utils.tissue_utils import get_tissue


import matplotlib.pyplot as plt
from skimage import io
import cv2

img_path = "/media/totem_disk/totem/ChengYu/FileZilla_rec/Masson tif/7/H1803134 1 masson/H1803134 1 masson_Wholeslide_默认_Extended.tif"
result_path = "/media/totem_disk/totem/ChengYu/Masson切图1/"
#讀取幻燈片
slide = Slide(img_path)
#掩碼級數爲2
mask_level = 0
svs_im = np.array(slide.get_thumb(mask_level))[:,:,:3]
#進行兩級下採樣
slide.get_level_downsample(mask_level)
#**2爲取平方
contour_area_threshold = 10000/(slide.get_level_downsample(mask_level)/slide.get_level_downsample(mask_level))**2
svs_regin_mask, _ = get_tissue(svs_im, contour_area_threshold)
print(svs_regin_mask.shape)
plt.imshow(svs_regin_mask)
plt.show()
#把有组织的坐标保存到data_lib[]中
data_lib,patch_C,patch_R = slide_sample_list(slide,svs_regin_mask,mask_level)
# print(patch_C)
# print(len(data_lib))
print("------------------------開始進行切割----------------------------")
#通过坐标进行patch大小的图片切割
for i in range(len(data_lib)):

    [cord, patch_index] = data_lib[i]
    img = slide.read_region(cord, 0, (patch_R, patch_C)).convert("RGB")

    img_array=np.array(img)[:,:,0:3]
    io.imsave(os.path.join(result_path+str(i) + 'more_blue.png'), img_array)


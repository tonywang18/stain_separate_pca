from __future__ import print_function

import os

import histomicstk as htk
from PIL import Image

from skimage import io
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
def PCA(inputImageFile):
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    Source = imInput
    # stains = ['hematoxylin',  # nuclei stain
    #           'eosin',  # cytoplasm stain
    #           'null']  # set to null if input contains only two stains

    I_0 = 255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map   # 获取库里面的染色矩阵，这里我们没用用到
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imInput, I_0)
    # print(w_est)
    # Perform color deconvolution  执行颜色的反卷积,得到細胞核跟細胞質以及背景三層通道
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)
    # print('Estimated stain colors (rows):', w_est.T[:3], sep='\n')    # xie
    #
    # Display results
    # for i in 0,1,2:
        # channel = htk.preprocessing.color_deconvolution.find_stain_index(stain_color_map[stains[i]], w_est)

        # print(deconv_result.Stains[:, :, i].shape)

        # print(type(deconv_result.Stains[:, :, i]))
        # io.imsave('/media/totem_disk/totem/ChengYu/PCA拆解结果/IHC拆解结果/混搭测试集/' + str(stains[i]) + ".png", deconv_result.Stains[:, :, i])

    # 以下代码就是将通过PCA方式获得的各个染色通道以及stain vector进行染色,deconv_result.Stains.shape[2]是通道数
    for c in range(deconv_result.Stains.shape[2]):  # input_image2.shape[2] = 3
        a = deconv_result.Stains[:, :, c].astype(float)  # a.shape = (256,256)
        if c == 0:
            rgb = a.reshape((-1), order='F')

        else:
            rgb = np.row_stack((rgb, a.reshape((-1), order='F')))
            # print(rgb)
    # 上面的代码就是将获得的3个通道进行叠加变形，成为一个(3，65526)的数组
    ref_vec = w_est[:3]
    d = rgb

    # print(rgb.shape)

    # 将范围变道0 - 1
    d = -np.log((d + 1) / 255)

    h_t = np.array([-ref_vec[0, 0]*d[0, :],
                    -ref_vec[1, 0]*d[0, :],
                    -ref_vec[2, 0]*d[0, :]])


    H = 255 * np.exp(h_t)
    H = H.T.reshape(imInput.shape, order='F')
    H = H * (H <= 255) * (H >= 0) + (H > 255) * 255

    e_t = np.array([-ref_vec[0, 1]*d[1, :],
                    -ref_vec[1, 1]*d[1, :],
                    -ref_vec[2, 1]*d[1, :]])

    E = 255 * np.exp(e_t)
    # print(E.shape)
    E = E.T.reshape(imInput.shape, order='F')
    E = E*(E <= 255)*(E >= 0) + (E > 255)*255
    # print(E.shape)

    return E, H, Source



images_dir = "/media/totem_disk/totem/fold_3验证集/images_orign/"  # H&E.png
out_dir = "/media/totem_disk/totem/ChengYu/PCA拆解结果/fold3_gray_invert"
images_list = os.listdir(images_dir)
count=0
images_list.sort(key=lambda x:int(x.split(".")[0]))
for i in tqdm(images_list):
    image_path = images_dir + i
    imInput = skimage.io.imread(image_path)[:, :, :3]
    Source = imInput
    image_name = i.split('.')[0]

    I_0 = 255
    #获取颜色矩阵
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(Source, I_0)
    #进行颜色反卷积得出反卷积结果
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(Source, w_est, I_0)
    #对取出的细胞核通道灰度图进行取反操作
    item_norm = (np.ones((256, 256)) * 255 - deconv_result.Stains[:, :, 1]).astype(np.uint8)
    #对图片进行保存
    io.imsave(os.path.join(out_dir,image_name + ".png"),item_norm)

    count+=1

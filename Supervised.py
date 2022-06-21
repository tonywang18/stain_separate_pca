'''=================================================
@Project -> File   ：KivenProject -> Supervised-new
@IDE    ：PyCharm
@Author ：Mr. kiven
@Date   ：2021/5/21 上午8:15
=================================================='''
from __future__ import print_function

import os

import histomicstk as htk
from histomicstk.preprocessing.color_deconvolution import stain_color_map
from skimage import io
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt
from tqdm import tqdm


def Supervised(inputImageFile):
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    Source = imInput
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    print('stain_color_map:', stain_color_map, sep='\n')

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains

    # create stain matrix
    # W = np.array([stain_color_map[st] for st in stains]).T

    # W = np.array([   # ID 2  HE
    #     [0.49015734, 0.04615336, 0],
    #     [0.76897085, 0.8420684, 0],
    #     [0.41040173, 0.5373925, 0],
    # ])
    # W = np.array([   # ID 3  IHC
    #     [0.650, 0.268, 0],
    #     [0.704, 0.570, 0],
    #     [0.286, 0.776, 0],
    # ])
    W = np.array([   #
         [0.20840239, 0.67617485, -0.65058866],
         [0.54785468, 0.71290365, 0.69618626],
         [0.81019979, 0.18589231, -0.3034124]
    ])




    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, W)

    input_image2 = imDeconvolved.Stains.astype(float).copy()  # input_image2.shape (256, 256, 3) 将图片进行复制，不改变原图


    for c in range(input_image2.shape[2]):  # input_image2.shape[2] = 3
        a = input_image2[:, :, c]   # a.shape = (256,256)
        if c == 0:
            rgb = a.reshape((-1), order='F')
        else:
            rgb = np.row_stack((rgb, a.reshape((-1), order='F')))
    # 上面的代码就是将获得的3个通道进行叠加变形，成为一个(3，65526)的数组
    ref_vec = W[:3]      # 注意这里 PCA这里进行了转置，但是ussupervised前面创建的时候就转置了，所以这里不用转置
    # print('ref_vec',ref_vec)

    d = rgb
    # 将范围变道0 - 1
    d = -np.log((d+1)/255)

    h_t = np.array([-ref_vec[0, 0]*d[0, :],
                    -ref_vec[1, 0]*d[0, :],
                    -ref_vec[2, 0]*d[0, :]])
    H = 255 * np.exp(h_t)
    H = H.T.reshape(imInput.shape, order='F')
    H = H*(H <= 255)*(H >= 0) + (H > 255)*255

    e_t = np.array([-ref_vec[0, 1]*d[1, :],
                    -ref_vec[1, 1]*d[1, :],
                    -ref_vec[2, 1]*d[1, :]])
    E = 255 * np.exp(e_t)
    E = E.T.reshape(imInput.shape, order='F')
    E = E*(E <= 255)*(E >= 0) + (E > 255)*255

    return H, E, Source


images_dir ="/media/totem_disk/totem/ChengYu/chaijie_test_HE/"  # H&E.png
out_dir = "/media/totem_disk/totem/ChengYu/chaijie_test_HE_result/"
images_list = os.listdir(images_dir)
# images_list = sorted(images_list, key=lambda x: int(x.split(".")[0]))
for i in tqdm(images_list):
    image_path = images_dir + i
    H, E, Source = Supervised(image_path)
    io.imsave(os.path.join(out_dir+'-H' + str(i)),H)
    io.imsave(os.path.join(out_dir +'-E' + str(i)),E)
    io.imsave(os.path.join(out_dir + '-Source' + str(i)), Source)



'''
inputImageFile = ('/media/totem_backup/totem/kiven/pannuke_image/fold_3/')  # H&E.png
images_list = os.listdir(inputImageFile)
for i in tqdm(range(len(images_list))):
    image = inputImageFile + images_list[i]
    H,E,Source= Supervised(image)

    # io.imsave(os.path.join('/media/totem_backup/totem/kiven/zuinew/Unsupervised/fold_3/' + str(i) + '.png'), Source)
    io.imsave(os.path.join('/media/totem_backup/totem/kiven/after/ThreeData/Supervised/fold_3/'+str(i)+'.png'),H)
    # io.imsave(os.path.join('/media/totem_backup/totem/kiven/after/徐哥给的IHC图片/PCA-IHC/'+str(i)+'-E.png'),E)

    # for i in range(0,3000):
    #     inputImageFile = ('/media/totem_backup/totem/kiven/pannuke_image/fold_3/'+str(i)+'.png')  # H&E.png
    #     H,E,Source= Supervised(inputImageFile)
    #     # io.imsave(os.path.join('/media/totem_backup/totem/kiven/项目的前期数据集的处理/zuinew/Supervised/fold_2/' + str(i) + '.png'), Source)
    #     io.imsave(os.path.join('/media/totem_backup/totem/kiven/after/ThreeData/Supervised/fold_3/'+str(i)+'.png'),H)
    #     # io.imsave(os.path.join('/media/totem_backup/totem/kiven/项目的前期数据集的处理/zuinew/Supervised/fold_2/' + str(i) + '-E.png'), E)
'''
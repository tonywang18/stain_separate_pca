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
from tqdm import tqdm



def Unsupervised(inputImageFile):
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    Source = imInput
    stains = ['hematoxylin',  # nuclei stain
              'eosin',  # cytoplasm stain
              'null']  # set to null if input contains only two stains

    W = np.array([stain_color_map[st] for st in stains]).T  # xie:
    # create initial stain matrix
    W_init = W[:, :2]


    # Compute stain matrix adaptively  自适应计算污点矩阵
    sparsity_factor = 0.5

    I_0 = 255
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0) # 将输入的RGB图像或矩阵“im_rgb”转换为SDA(着色黑暗)空间的颜色反褶积。

    W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
        im_sda, W_init, sparsity_factor,
    )

    # perform sparse color deconvolution  执行稀疏颜色反卷积
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
        imInput,
        htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
        I_0,
    )

    # print('Estimated stain colors (in rows):', W_est.T, sep='\n')

    for c in range(imDeconvolved.Stains.shape[2]):  # input_image2.shape[2] = 3
        a = imDeconvolved.Stains[:, :, c].astype(float)# a.shape = (256,256)
        if c == 0:
            rgb = a.reshape((-1), order='F')
        else:
            rgb = np.row_stack((rgb, a.reshape((-1), order='F')))
    # 上面的代码就是将获得的3个通道进行叠加变形，成为一个(3，65526)的数组
    ref_vec = W_est[:3]
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

images_dir ="/media/totem_disk/totem/ChengYu/切割好的圖片/"  # H&E.png
out_dir = "/media/totem_disk/totem/ChengYu/无监督拆解结果/"
images_list = os.listdir(images_dir)
# images_list = sorted(images_list, key=lambda x: int(x.split(".")[0]))
for i in tqdm(images_list):
    image_path = images_dir + i
    E, H, Source = Unsupervised(image_path)
    image_name = i.split('.')[0]
    io.imsave(os.path.join(out_dir+"H.png"),H)
    io.imsave(os.path.join(out_dir +"E.png"), E)



'''
inputImageFile = ('/media/totem_backup/totem/kiven/after/徐哥给的IHC图片/aa/')  # H&E.png
images_list = os.listdir(inputImageFile)
for i in tqdm(range(len(images_list))):
    image = inputImageFile + images_list[i]
    print(image)
    H,E,Source= Unsupervised(image)

    # io.imsave(os.path.join('/media/totem_backup/totem/kiven/zuinew/Unsupervised/fold_3/' + str(i) + '.png'), Source)
    io.imsave(os.path.join('/media/totem_backup/totem/kiven/after/徐哥给的IHC图片/aa-aa/'+str(i)+'.png'),H)

    #io.imsave(os.path.join('/media/totem_backup/totem/kiven/after/徐哥给的IHC图片/Pytorch-lear/Colon_CD3_40X-拆解/' + str(i) + '-E.png'), E)
    #

    # inputImageFile = ('/media/totem_backup/totem/kiven/pannuke_image/fold_1/'+str(i)+'.png')  # H&E.png
    # H, E, Source= Unsupervised(inputImageFile)
    # io.imsave('image/' + str(i) + '.png', Source)
    # io.imsave('image/'+str(i)+'-H.png',H)
    # io.imsave('image/' + str(i) + '-E.png', E)

'''
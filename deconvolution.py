# -*- coding:utf-8 -*-
"""
Created on Thu Nov 22 17:38:04 2018

@author: Bohrium.Kwong
"""

import numpy as np
import cv2
import math
from skimage import io, exposure, img_as_float
import warnings
warnings.filterwarnings("ignore")



def saparatestains(input_image,stainColorMap):
    he = np.array(stainColorMap['hematoxylin'])
    dab = np.array(stainColorMap['dab'])
    res = np.array(stainColorMap['res'])

    HDABtoRGB = np.array([he/np.linalg.norm(he, axis=0, keepdims=True),
                      dab/np.linalg.norm(dab, axis=0, keepdims=True),
                     res/np.linalg.norm(res, axis=0, keepdims=True)])
    RGBtoHDAB = np.linalg.inv(HDABtoRGB)


    input_image2 = input_image.astype(float) + 2
    input_image2 = -np.log(input_image2)
    imageOut = np.dot(input_image2.reshape((-1, 3)), RGBtoHDAB)
    imageOut = imageOut.reshape(input_image.shape)

    for c in range(imageOut.shape[2]):
        imageOut[:, :, c] = (imageOut[:, :, c] - imageOut[:, :, c].min()) \
                            / (imageOut[:, :, c].max()-imageOut[:, :, c].min())
        imageOut[:, :, c] = 1 - imageOut[:, :, c]
        v_min, v_max = np.percentile(imageOut[:, :, c], (0.1, 98.991))
        imageOut[:, :, c] = exposure.rescale_intensity(imageOut[:, :, c], in_range=(v_min, v_max))

    return imageOut


def ihc_sep(input_image,stainColorMap):
    he = np.array(stainColorMap['hematoxylin'])
    dab = np.array(stainColorMap['dab'])
    res = np.array(stainColorMap['res'])

    Io = 255
    input_image2 = input_image.astype(float).copy()
    print('input_image2',input_image2)
    print('input_image2.shape',input_image2.shape)

    for c in range(input_image2.shape[2]):
        a = input_image2[:, :, c]
        if c == 0:
            rgb = a.reshape((-1), order='F')
        else:
            rgb = np.row_stack((rgb, a.reshape((-1), order='F')))

    print('rgb.shape',rgb.shape)
    print('rgb',rgb)      # rbg = m 只是rgb是float m是整数

    od_m = -np.log((rgb+1)/255)
    # sda_fwd[[8.49854976  8.49854976  7.42581155... 14.02739698 14.02739698  14.51957691]
    print('od_m',od_m)
    print('od_m',od_m.T)
    print('od_m.shape.T',od_m.T.shape)
    print('od_m.shape',od_m.shape)



    ref_vec = np.array([he, dab, res]).T

    d = np.dot(np.linalg.inv(ref_vec), od_m)
    print('d',d)
    print('d.shape',d.shape)



    h_t = np.array([-ref_vec[0, 0]*d[0, :],
                -ref_vec[1, 0]*d[0, :],
                -ref_vec[2, 0]*d[0, :]])

    H = 255 * np.exp(h_t)
    H = H.T.reshape(input_image2.shape, order='F')
    H = H*(H <= 255)*(H >= 0) + (H > 255)*255

    e_t = np.array([-ref_vec[0, 1]*d[1, :],
                -ref_vec[1, 1]*d[1, :],
                -ref_vec[2, 1]*d[1, :]])

    E = 255 * np.exp(e_t)
    E = E.T.reshape(input_image2.shape, order='F')
    E = E*(E <= 255)*(E >= 0) + (E > 255)*255

    return H, E

if __name__ == '__main__':
    stainColorMap = {
    'hematoxylin': [ 0.17793963 , 0.79579523 , 0.57883283],
    'dab':         [ 0.53433727 , 0.72663972 , 0.43183145],
    'res':         [-0.20035128  ,0.60519727 ,-0.77045157]
}
    im_input = io.imread('/media/totem_backup/totem/kiven/pannuke_image/fold_1/0.png')
    H,color_dec_img = ihc_sep(im_input,stainColorMap)
    io.imsave('H.png', H)
    io.imsave('E.png', color_dec_img)
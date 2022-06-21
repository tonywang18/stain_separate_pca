import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
# 对images进行提取
path = r'D:\数据集\PanNukeDatasetOriginalValid\fold_3\images.npy'# 要转换为图片的.npy文件
result_path=r'D:\数据集\fold3_images\\'
data = np.load(path)
pp=np.array(data)
for i in range(data.shape[0]):
    img_array=pp[i,:,:,0:3]
    io.imsave(os.path.join(result_path+str(i)+".png"),img_array)

# # 对mask进行提取
# path = '/media/totem_disk/totem/PanNukeDataset/PanNukeDatasetOriginalTrain/fold_1/masks.npy'# 要转换为图片的.npy文件
# result_path="/media/totem_disk/totem/ChengYu/pannuke数据集/fold1_masks/"
# data = np.load(path)
# pp=np.array(data)
# print(pp.shape)

# #对types进行提取
# path = '/media/totem_disk/totem/PanNukeDataset/PanNukeDatasetOriginalTrain/fold_1/types.npy'# 要转换为图片的.npy文件
# result_path="/media/totem_disk/totem/ChengYu/pannuke数据集/fold1_masks/"
# data = np.load(path)
# print(data.shape)
# pp=np.array(data)
# # for i in range(data.shape[0]):
# #     img_array=pp[i,:,:,0:3]
# #     io.imsave(os.path.join(result_path+str(i)+".png"),img_array)
# for i in range(len(data)):
#     print(data[i])
# # list=set(data)
# # print(list)
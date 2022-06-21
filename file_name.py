import os

img_path = "/media/totem_disk/totem/Lung_HE_20220401/Set2_transform/HE/"
result_path = "/media/totem_disk/totem/Wnc-Projects/Openslide-Learn/result/"

for root,dirs,files in os.walk(img_path):

    for i in range(len(files)):
        if ".svs" in files[i]:
            os.makedirs(files[i].split("f")[0])
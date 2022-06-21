import numpy as np




def slide_sample_list(slide,svs_regin_mask,mask_level=2,model_input_shape = 256,step_length = 2,get_mpp = True,patch_R = 256,patch_C = 256):
    """
    generate the cord list which to be predicted
    :param slide: svs file read by Slide
    :param model_input_shape: segmentation model input & output size
    :step_length: the reciprocal value of length of step in the  patch for-loop of svs
    :get_mpp: default value of the patch_size in processing or not
    :return:
    """
    slide_width, slide_height = slide.get_level_dimension(0)
    if get_mpp:
        try:
            if round(slide.get_mpp()/0.00025) == 1:
                patch_R = 512
                patch_C = 512
            elif round(slide.get_mpp()/0.00025) == 2:
                patch_R = 256
                patch_C = 256
            else:
                patch_R = 128
                patch_C = 128
        except Exception:
            print(' mpp error!')
        # 乳腺癌区域分割模型共有5类，是基于20X下以256×256采样的图像训练的，所以需要根据倍率进行不同的调整
    # step_length = 2 half step
    w_count = slide_width // round(patch_C / step_length)
    h_count = slide_height // round(patch_R / step_length)

    level_downsamples = slide.get_level_downsample(level=mask_level)

    data_lib = []

    for w in range(w_count):
        for h in range(h_count):
            w_cord = w * round(patch_C / step_length)
            h_cord = h * round(patch_R / step_length)
            bottom = round(h_cord / level_downsamples)
            top = bottom + round(patch_R / level_downsamples) -1
            left = round(w_cord / level_downsamples)
            right = left + round(patch_C / level_downsamples) -1


            if np.sum(svs_regin_mask[bottom : top,left : right ] > 0) > 0.6 * (patch_R * patch_C / level_downsamples / level_downsamples):
                data_lib.append([(w_cord , h_cord),(w,h)])

    return data_lib,patch_C,patch_R
    # slide_width, slide_height = slide.get_level_dimension(0)
    # if get_mpp:
    #     try:
    #         if round(slide.get_mpp()/0.00025) == 1:
    #             patch_R = patch_R * 2
    #             patch_C = patch_C * 2
    #         elif round(slide.get_mpp()/0.00025) == 2:
    #             patch_R = patch_R
    #             patch_C = patch_C
    #         else:
    #             patch_R = round(patch_R / 2)
    #             patch_C = round(patch_C / 2)
    #     except Exception:
    #         print(' mpp error!')
    #     # 乳腺癌区域分割模型共有5类，是基于20X下以256×256采样的图像训练的，所以需要根据倍率进行不同的调整
    # # step_length = 2 half step
    # w_count = slide_width // round(patch_C / step_length)
    # h_count = slide_height // round(patch_R / step_length)
    # level_downsamples = slide.get_level_downsample(level=mask_level)
    #
    # data_lib = []
    #
    # for w in range(w_count):
    #     for h in range(h_count):
    #         w_cord = w * round(patch_C / step_length)
    #         h_cord = h * round(patch_R / step_length)
    #         bottom = round(h_cord / level_downsamples)
    #         top = bottom + round(patch_R / level_downsamples) -1
    #         left = round(w_cord / level_downsamples)
    #         right = left + round(patch_C / level_downsamples) -1
    #         if np.sum(svs_regin_mask[bottom : top,left : right ] > 0) > 0.6 * (patch_R * patch_C / level_downsamples / level_downsamples):
    #             data_lib.append([(w_cord , h_cord),(w,h)])
    #
    # return data_lib,patch_C,patch_R


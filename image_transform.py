import tensorflow as tf
from PIL import Image
import numpy as np

from AI.freez_to_graph import build_net


def image_transform(image,frozen_model,trans_intensity=.3):
    '''
    this function gets an image, a filter and the intensity ratio and produce the styled image
    size of image should not be too big
    :param image: the input image address, given by the user
    :param frozen_model: the filter name. which is the name of the folder. which is saved file in filter/"filter name'/frozen_model.pb
    :param trans_intensity: the intesity that the filter applies to image. a float between 0 and 1 . with 0 the patterns by the network is minimum and with one nothing from original image will remain
    :return: the styled image. a PIL image object can be handeled with PIL

    example:
    img=image_transform('arian.jpg','color',.3)

    '''
    def resize_aspect(img,final_w):
        '''
        resizing image but keeping the aspect ratio
        :param img:
        :param final_w:
        :return:
        '''

        w,h=img.size

        ratio=final_w/w
        new_h=int(h*ratio)
        new_w=int(w*ratio)
        img=img.resize((new_w,new_h))
        return img



    frozen_model='AI\\filters\\'+frozen_model+'\\frozen_model.pb'
    predict,x_f,sess,ratio=build_net(frozen_model)
    a = Image.open(image)
    if a.size[0]>1024:
        a=resize_aspect(a,512)

    # a = a.resize((512,512))# if resize was ever needed
    a = np.array(a, np.uint8)
    a_shape = np.array(a.shape)
    a = np.reshape(a, [1, a_shape[0], a_shape[1], a_shape[2]]) # preparing to acceptable format for network

    Predict = sess.run(predict, feed_dict={x_f: a, ratio: [trans_intensity]})

    Predict = np.array(Predict, np.uint8)
    pred_shape = np.array(Predict.shape)
    Predict = np.reshape(Predict[0], pred_shape[1:4])# bring back to original form

    img = Image.fromarray(Predict)
    return img


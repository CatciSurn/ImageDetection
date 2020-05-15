import base64
import gc
import time

import cv2
import numpy
import redis
import json
from io import BytesIO
from PIL import Image
from detector import DetectorRetinaface, DetectorDBFacce, DetectorGapMobileNetV3_SSD, DetectorYoloV3, DetectorCrnn


load_models_once = True
if load_models_once:
    detectorRetinaface = DetectorRetinaface()
    detectorDBFacce = DetectorDBFacce()
    detectorGapMobileNetV3_SSD = DetectorGapMobileNetV3_SSD()
    detectorYoloV3 = DetectorYoloV3()
    detectorCrnn = DetectorCrnn()



def resize_img(image, max_length=500):
    ori_height, ori_width = image.shape[:2]

    rate = max([ori_height, ori_width])/max_length

    if rate > 1:
        img = cv2.resize(image, (0, 0), fx=1/rate, fy=1/rate, interpolation=cv2.INTER_NEAREST)  # 最近邻插值法缩放
        return img, rate
    else:
        return image, 1


r = MyRedis()

while True:
    text_crnn = 'none_text_crnn'
    location_list = 'none_location_list'
    interval = 0
    detect_type = 'none_detect_type'
    image_name = 'none_mage'
    rate = 1
    image_path = ''
    try:
        image = cv2.imread(image_path)

        location_list = []
        interval = 0
        rate = 1


        image, rate = resize_img(image, max_length=320)
        image = image[..., ::-1]

        detect_instance = DetectorGapMobileNetV3_SSD()



        location_list = numpy.round(numpy.asarray(location_list) * rate).tolist()


        data_push = {
            'location_list': location_list,
            'interval': interval,
            'detect_type': detect_type
        }
        print('检测成功，结果是：', data_push)

    except Exception as e:
        print('发生异常:', e.args)
        data_push = {
            'location_list': '',
            'interval': '',

            'succsess': 0,

            'detect_type': detect_type
        }







import cv2
import numpy

from mobilenetv3_ssd.vision.utils.misc import Timer
import os
import torch
from mobilenetv3_ssd.vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor


class DetectorGapMobileNetV3_SSD():
    def __init__(self, device='cpu'):
        '''

        :param device: cpu or cuda
        '''

        # 请设置路径，不清楚相对路径请写绝对路径
        model_path = 'mobilenetv3_ssd/weights/mb3-ssd-lite-Epoch-490-Loss-0.3229829967021942.pth'
        label_path = 'mobilenetv3_ssd/weights/labels.txt'


        print(os.curdir, "****", os.getcwd())

        class_names = [name.strip() for name in open(label_path).readlines()]
        if device=='cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True, device=device)
        net.load(model_path)
        net.to(torch.device(device))
        self.predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=10, device=device)
        self.timer = Timer()

    def detect(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.timer.start()
        boxes, labels, probs = self.predictor.predict(image, 10, 0.4)
        interval = self.timer.end()
        result = boxes.numpy().tolist()

        return (result, interval)

if __name__ == '__main__':

    filepath_gap = "image.jpg"
    image_gap = cv2.imread(filepath_gap)

    detector = DetectorGapMobileNetV3_SSD()

    bboxes,interval  = detector.detect(image_gap)
    print('检测浮点数结果:', bboxes)
    bboxes = numpy.round(bboxes).astype(numpy.uint16)
    print('检测整型结果:', bboxes)
    print('检测时间消耗：', interval)


    for b in bboxes:


        cv2.rectangle(image_gap, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
    cv2.imwrite('DetectorYoloV3_{}'.format(filepath_gap), image_gap)
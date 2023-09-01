# Yolov7_darknet_opencv_dnn

<div align="center">
  <img width="80%" src="result/demo/bus_result.jpg">
</div>

## Description

Yolov7 use darknet with OpenCV dnn

source code - Pytorch (use to reproduce results):https://github.com/WongKinYiu/yolov7

paper:https://arxiv.org/abs/2207.02696

model:https://github.com/AlexeyAB/darknet/issues/8595

| Darknet | Opencv_dnn |
| ------------- |:-------------:|
|  <div align="center"><img width="80%" src="result/demo/darknet_predictions.jpg"></div>     |   <div align="center"> <img width="80%" src="result/demo/opencv_dnn_predictions.jpg"></div> |


## Getting Started

### Environment
* Ubuntu 20.04
* python 3.8
* OpenCV 4.x up

### Installing

```
git clone https://github.com/aa0308qq/dnn_darknet.git
python3 model.py
```

### Executing program

* It is python library
```
import dnn_darknet
model=dnn_darknet.model.Yolov7(cfg_path,weights_path,class_name_path)
img=cv2.imread(img_path)
class_names,confidences,boxes=model.detect(img)
inference_img=model.draw_boxes(image,class_names,confidences,boxes)
cv2.imshow('result',inference_img)
cv2.waitKey(0)
```

## Reference

* Yolov7:https://github.com/WongKinYiu/yolov7
* Yolov5:https://github.com/ultralytics/yolov5/tree/master
* darknet:https://github.com/AlexeyAB/darknet


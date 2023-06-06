import os
import cv2
import time
import numpy as np
try:
    from . import utils
except:
    import utils
from threading import RLock

class Yolov7():
    def __init__(self,cfg_path:str,weights_path:str=None,cls_names_path:str=None,input_shape:list =[416,416,3]):
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')
        if not os.path.isfile(cfg_path) and not os.path.isfile(weights_path) and not os.path.isfile(cls_names_path):
            cfg_path=os.path.join(model_path,f'{cfg_path}')
            weights_path=os.path.join(model_path,f'{weights_path}')
            cls_names_path=os.path.join(model_path,f'{cls_names_path}')
        
        if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path):
            raise Exception("no model")
        if not os.path.isfile(cls_names_path):
            raise Exception("no class names")
        
        self.names=utils.get_cls_names(cls_names_path)
        self.input_shape=np.array(input_shape,dtype=np.int16)
        self.net=cv2.dnn.readNetFromDarknet(cfg_path,weights_path)    

        if cv2.cuda.getCudaEnabledDeviceCount()> 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.output_layers=np.array(self.get_output_layers(self.net))
        
        self.model_lock=RLock()
        self.model_lock.acquire()
        test_frame=np.zeros(self.input_shape,dtype=np.uint8)
        check=time.time()
        _,_,_=self.detect(test_frame)
        print(f'warmup_time:{time.time()-check}')
        self.model_lock.release()

    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    def detect(self,frame,class_names=None,confidences=None,boxes=None):
        self.model_lock.acquire()
        blob=utils.preprocess(frame,[self.input_shape[1],self.input_shape[0]])
        self.net.setInput(blob)
        pred = self.net.forward(self.output_layers)
        start=time.time()
        post_pred=utils.postprocess(pred,agnostic=True)
        print(f'postprocess:{time.time()-start}')
        if len(post_pred)>0:
            class_names=self.names[post_pred[:,5].astype(np.int16)]
            confidences=post_pred[:,4]
            boxes=post_pred[:,:4]
        self.model_lock.release()
        return class_names,confidences,boxes
    
    def draw_boxes(self,frame=None,class_names=None,confidences=None,boxes=None):
        if frame is None or class_names is None or confidences is None or boxes is None:
            return frame
        img=frame.copy()
        img_height,img_width=img.shape[:2]
        for class_name,confidence,box in zip(class_names,confidences,boxes):
            x1=int(box[0]*img_width)
            y1=int(box[1]*img_height)
            x2=int(box[2]*img_width)
            y2=int(box[3]*img_height)
            font_scale=utils.get_optimal_font_scale(x2-x1,f'{str(class_name)}:{str(confidence)}',cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(img, f'{str(class_name)}:{str(confidence)}',(x1,y1), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img,(x1,y1), (x2,y2), (0, 0, 255), 1)
        return img

if __name__ == "__main__":
    yolov7_model=Yolov7('yolov7-tiny.cfg','yolov7-tiny.weights','coco.names')
    image=cv2.imread('/home/joe/workspace/Joe/yolov7_opencv_dnn/dnn_darknet/data/bus.jpg')
    start_inference=time.time()
    class_names,confidences,boxes=yolov7_model.detect(image)
    print(f'inference_time:{time.time()-start_inference}')
    start=time.time()
    inference_img=yolov7_model.draw_boxes(image,class_names,confidences,boxes)
    print(f'draw_time:{time.time()-start}')
    cv2.imshow('result',inference_img)
    cv2.waitKey(0)
   
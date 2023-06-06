import cv2
import numpy as np

def get_cls_names(cls_names_path):
    with open(cls_names_path, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    while names[-1]=='':
        names.pop()
    return np.array(names)

def preprocess(frame,input_shape):
    blob = cv2.dnn.blobFromImage(frame,1/255, input_shape,swapRB=True)
    return blob

def postprocess(pred,conf_thres=0.25,nms_thres=0.65,max_nms=30000,agnostic=False,min_wh=0,max_wh=4096):
    pred=np.vstack(pred)
    nc = pred.shape[1] - 5
    multi_label = nc > 1
    pred=pred[pred[:,4]>conf_thres]
    boxes=xywh_xyxy(pred[:, :4])
    confidences=np.max(pred[:, 5:], axis=1)
    classes=np.argmax(pred[:, 5:], axis=1)
    confidence_candidates=confidences>conf_thres
    confidences=confidences.reshape(-1,1)
    classes=classes.reshape(-1,1)
    # Detections matrix nx6 (xyxy, conf, cls)
    result=np.concatenate((boxes,confidences,classes),axis=1)[confidence_candidates]
    if not pred.shape[0]:  # no boxes
        return []
    elif pred.shape[0] > max_nms:
        result= result[result[:, 4].argsort()[:max_nms]]
    scores_bias=result[:, 5:6] * (0 if agnostic else max_wh)
    nms_boxes,nms_scores = result[:, :4]+scores_bias, result[:, 4]
    nms_index = cv2.dnn.NMSBoxes(nms_boxes.tolist(), nms_scores.tolist(), conf_thres, nms_thres)
    result=result[nms_index]
    result[result<0]=0
    return result
            
def xywh_xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def get_optimal_font_scale(width,text,fontface):
    for scale in range(59,10,-1):
        textSize = cv2.getTextSize(text, fontFace=fontface, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width < width*0.8):
            return scale/10
    return 1

if __name__=="__main__":
    result=get_cls_names('/home/joe/workspace/yolov7/coco.names')
   

# object_detection.py
# Uses ultralytics YOLO if available; otherwise fallback to OpenCV DNN with YOLOv4-tiny or MobileNet-SSD
import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# default class names: coco
COCO_NAMES = None
def load_coco_names(path):
    global COCO_NAMES
    if COCO_NAMES is not None:
        return COCO_NAMES
    if COCO_NAMES is None:
        path = os.path.join(os.path.dirname(__file__), 'assets', 'coco.names')
    if not os.path.exists(path):
        raise FileNotFoundError(f'COCO names file not found at {path}')

        # try to create default COCO list (80 classes)
        COCO_NAMES = [
            'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
            'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
            'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
            'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
            'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
            'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
            'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
            'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]

    else:
        with open(path, 'r') as f:
            COCO_NAMES = [l.strip() for l in f.readlines()]
    return COCO_NAMES

# Ultralytics YOLO wrapper
class YOLODetector:
    def __init__(self, model_path=None, conf=0.4, device='cpu'):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError('ultralytics package is not available')
        if model_path is None:
            # use yolov8n (nano) if ultralytics installed and internet allowed; else require path
            model_path = 'yolov8n.pt'
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame):
        """
        frame: BGR OpenCV image
        returns list of (box [x,y,w,h], classID, confidence)
        """
        # ultralytics expects RGB
        results = self.model(frame[:, :, ::-1], imgsz=640, conf=self.conf, device=self.device)[0]
        detections = []
        for r in results.boxes:
            # r.xywh, r.conf, r.cls
            box = r.xywh[0].tolist()
            # convert center xywh to x,y,w,h (top-left)
            cx, cy, w, h = box
            x = float(cx - w/2); y = float(cy - h/2)
            cls = int(r.cls[0].item()) if hasattr(r, 'cls') else int(r.cls)
            conf = float(r.conf[0].item()) if hasattr(r, 'conf') else float(r.conf)
            detections.append(([x, y, w, h], cls, conf))
        return detections

# Fallback using OpenCV DNN YOLOv4-tiny loader (requires cfg and weights)
def load_yolov4_tiny(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    classes = load_coco_names(names_path)
    return net, classes

def yolo_v4_tiny_detect(net, classes, frame, conf_thresh=0.4, nms_thresh=0.4, input_size=(416,416)):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getLayerNames()
    out_names = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(out_names)
    H, W = frame.shape[:2]
    boxes, confidences, classIDs = [], [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = float(scores[classID])
            if conf > conf_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(conf)
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append((boxes[i], classIDs[i], confidences[i]))
    return results

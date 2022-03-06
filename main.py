from cvlib.object_detection import YOLO
import cv2

weights="yolov3.weights"
config="yolov3.cfg"
labels="coco.names"
img=cv2.imread("img.jpg")
img=cv2.resize(img,(1280,960))
yolo = YOLO(weights, config,labels)
bbox, label, conf = yolo.detect_objects(img)
img1=yolo.draw_bbox(img, bbox, label, conf)
cv2.imshow("img1",img)
cv2.waitKey(0)

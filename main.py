from cvlib.object_detection import YOLO
import cv2

weights="/home/pi/darknet/yolov3.weights"
config="/home/pi/darknet/cfg/yolov3.cfg"
labels="/home/pi/darknet/data/coco.names"
img=cv2.imread("3.jpg")
img=cv2.resize(img,(640,480))
yolo = YOLO(weights, config,labels)
bbox, label, conf = yolo.detect_objects(img)
img1=yolo.draw_bbox(img, bbox, label, conf)
cv2.imshow("img1",img)
cv2.waitKey(0)
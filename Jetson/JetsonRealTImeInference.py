import tensorflow as tf
import numpy as np
import cv2
import cvzone
import time

from ultralytics import YOLO
from mmap import ACCESS_DEFAULT
from ultralytics.yolo.utils.plotting import Annotator


def preprocess_image(img_array):
  img = tf.image.resize(img_array, (IMAGE_WIDTH,IMAGE_HEIGHT))
  img = img/255
  return img


###### Parameters ######
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LOW_LIGHT_MODEL_PATH = r'Model_Files/vgg_tflite_dynamicQ.tflite'

YOLO_PATH = r'Model_Files/Yolov8_LL.pt'

###### Set up Low Light TFLite Model ######
interpreter = tf.lite.Interpreter(model_path=LOW_LIGHT_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

###### YOLO Set up ######
model_yolo = YOLO(YOLO_PATH)

##### Process Video Stream ######
vid = cv2.VideoCapture(0)


while(True):
    t1 = time.time()
    ret, frame = vid.read()
    frame = cv2.resize(frame,[IMAGE_WIDTH,IMAGE_HEIGHT])

    ################ LOW LIGHT MODEL ####################
    t2 = time.time()
    process_frame = preprocess_image(frame)
    process_frame = np.array(process_frame)
    input_data = np.expand_dims(process_frame, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    LL_RGB_frame = np.array((output_data[0]*255).astype(np.uint8))
    t3 = time.time()

    ################# YOLO INFERENCE #######################
    yolo_preds = model_yolo(source=LL_RGB_frame, conf=0.25)

    for r in yolo_preds:
        annotator = Annotator(LL_RGB_frame.copy())
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  
            c = box.cls
            annotator.box_label(b, model_yolo.names[int(c)])
            #annotator.box_label(b)
        
    yolo_img = annotator.result()
    t4 = time.time()
    print(f"t_LL : {t3-t2}  t_YOLO : {t4-t3}")
    ## FPS
    fps  = 1 / (t4 - t1)
    print(f'FPS RATE : {fps}')
    cv2.putText(frame, str(fps), (20,30), cv2.FONT_HERSHEY_COMPLEX, 
                   1, (0,255,0), 2, cv2.LINE_AA)

    ## Display frames
    # print(frame.shape,LL_RGB_frame.shape,yolo_img.shape)
    imgList = [frame, LL_RGB_frame, yolo_img]
    stackedImg = cvzone.stackImages(imgList, 3, 0.6)
    cv2.imshow('frame', stackedImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()


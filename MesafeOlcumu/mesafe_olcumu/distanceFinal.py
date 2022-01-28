# Baran_Akbay
import cv2 
import numpy as np

KNOWN_DISTANCE = 45
PERSON_WIDTH = 16 
MOBILE_WIDTH = 3.0 

# nesne algılama sabitlerini aldım.
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# renk tanımlamaları yaptım
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# yazı tiplerini tanımladım 
FONTS = cv2.FONT_HERSHEY_COMPLEX

# classes'den isimleri çektim. 
class_names = []
with open("/home/brx/PycharmProjects/MesafeOlcumu/mesafe_olcumu/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv2.dnn.readNet('/home/brx/PycharmProjects/MesafeOlcumu/mesafe_olcumu/kucukNesne.weights', '/home/brx/PycharmProjects/MesafeOlcumu/mesafe_olcumu/kucukNesne.cfg')

yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# nesne tanımlama fonksiyonu
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # nesne eklemek için bir boş liste.
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # sınıf kimliğine göre renk atama  
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # tanımlanan nesneyi dikdörtgen içine aldım
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # verileri çekiyorum 
 
        if classid ==0: 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==67:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
         
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# mesafe ölçme fonksiyonu 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# kayıtlı resimleri okuma  
ref_person = cv2.imread('/home/brx/PycharmProjects/MesafeOlcumu/mesafe_olcumu/ReferenceImages/image12.png')
ref_mobile = cv2.imread('/home/brx/PycharmProjects/MesafeOlcumu/mesafe_olcumu/ReferenceImages/image2.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# odak uzaklığını hesaplama  
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        cv2.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv2.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


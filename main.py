import ultralytics
print(ultralytics.__version__)

import cv2
import numpy as np
from sort.sort import Sort 
from utils import get_car,read_license_plate, write_csv
from ultralytics import YOLO


results = {}
mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model for vehicle detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Pre-trained YOLOv8 model for license plate detection

# load video
cap = cv2.VideoCapture('./sample.mp4')
vehicles = [2, 3, 5, 7]

# read frames
ret = True
frame_nmr = -1
while ret:  
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 100:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]  # first element contains the results
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car  
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1 :
        
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # process license plate
                # easyOCR or any other OCR model
                # apply filters to improve OCR accuracy like grayscale, thresholding, etc.
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                )

                # display crops and thresholded images
                #cv2.imshow('original_crop', license_plate_crop)
                #cv2.imshow('thresh_crop', license_plate_crop_thresh)
                #cv2.waitKey(0)

                # read license plate number
                license_plate_text, confidence = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None : 
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2] } ,
                                                'license_plate': { 'bbox': [x1, y1, x2, y2] ,
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': confidence}}
            
# write results on frame
write_csv(results, './main.csv') 
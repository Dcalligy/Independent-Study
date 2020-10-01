                                #####################
                                #                   #
                                # ABOUT THE PROJECT #
                                #                   #
                                #####################
# python faceDetect.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel                               
# A Gender and Age Detection program that uses Deep Learning to accurately identify the gender of a person from a single images of a face.
# The predicted gender may be one of 'Male' and 'Female', and the predicted age may be one of the following ranges -
# (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer).
# Finding the exact age can be trickey because you need to factor things in like makeup, lighting, obstructions, and facial expressions

import cv2
import math
import argparse
#from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import argparse
from imutils import face_utils
import imutils
import time
import dlib
import sys
import os

"""
The contents of the Project:
----------------------------

   * opencv_face_detector.pbtxt
   * opencv_face_detector_uint8.pb
   * age_deploy.prototxt
   * age_net.caffemodel
   * gender_deploy.prototxt
   * gender_net.caffemodel
   * a few pictures to try the project on
   * detect.py

To detect the users face we use a protobuf file (protocol buffer), this file contains and runs the trained model. The .pb file (protobuff) holds the protofbut in binary format, one file is written in text format. The files listed in the table of contents are known as TensorFlow files. Detecting the age and gender is done with the .prototxt file, which describes the network configuration and the .caffemodel file defines the internal state of the parameters layers.
"""

def faceScan(net, frame, conf_threshold=0.7):
    
    startFrame = frame.copy()
    frameHeight = startFrame.shape[0]
    frameWidth = startFrame.shape[1]
    blob = cv2.dnn.blobFromImage(
        startFrame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    userFound = net.forward()
    userBoxes = []
    
    # Creates a rectangular frame that is used to detect a users face
    # face dimensions 
    for i in range(userFound.shape[2]):
        confidence = userFound[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(userFound[0, 0, i, 3] * frameWidth)
            y1 = int(userFound[0, 0, i, 4] * frameHeight)
            x2 = int(userFound[0, 0, i, 5] * frameWidth)
            y2 = int(userFound[0, 0, i, 6] * frameHeight)
            userBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(startFrame, 
                          (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return startFrame,userBoxes



# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
parser.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
parser.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
args = vars(parser.parse_args())

"""
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
"""
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

userNet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
ageNet = cv2.dnn.readNetFromCaffe(
    "model_detectors/age_deploy.prototxt",
    "model_detectors/age_net.caffemodel"
)
genderNet = cv2.dnn.readNetFromCaffe(
    "model_detectors/gender_deploy.prototxt",
    "model_detectors/gender_net.caffemodel"
)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
padding = 20

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg, userBoxes = faceScan(userNet, frame)

    if not userBoxes:
        print("No face detected! Please look into the camera!!!")
        #time.sleep(3.0)
    
    # Scan for user's face, read the age and gender net caffe models and display the results near the facebox
    for userBox in userBoxes:
        face = frame[max(0, userBox[1] - padding):
                   min(userBox[3] + padding, frame.shape[0] - 1), max(0, userBox[0] - padding)
                   :min(userBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB = False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        cv2.putText(resultImg, f'{gender}, {age}', (userBox[0], userBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    # show the frame around the user
    cv2.imshow("Detecting age and gender", resultImg)
    key = cv2.waitKey(1) & 0xFF # used to terminate app. key = q 
    
    if key == ord("q"):
        break

# display results
print(f'Gender: {gender}')
print(f'Age: {age[1:-1]} years')

# release the hounds
video.release()
cv2.destroyAllWindows()

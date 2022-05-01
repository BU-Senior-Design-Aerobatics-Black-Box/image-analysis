#! /usr/bin/env python3

import cv2
import time
import datetime
import matplotlib.pyplot as plt
import time

from get_angle import get_angle

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(2)
cam1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cam2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

# get rid of buffer so we always grab the latest frame
cam1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_width1 = int(cam1.get(3))
frame_height1 = int(cam1.get(4))
frame_width2 = int(cam2.get(3))
frame_height2 = int(cam2.get(4))

frame_size1 = (frame_width1, frame_height1)
frame_size2 = (frame_width2, frame_height2)
writer1 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test1_analysis.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size1)
writer2 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test2_analysis.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size2)
# count = 0
while True:
    tick = time.time()
    #ret1, image1 = cam1.read()
    cam1.grab()
    ret1, image1 = cam1.retrieve()
    ret2, image2 = cam2.read()
    tock = time.time()
    print(f"cam.read() took {tock-tick}")
    #hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    #out.write(hsv)
    image_type = type(image1)
    #print(f"image type: {type(image)}")
    print("image type", image_type)
    print("image shape", image1.shape)
    tick = time.time()
    if ret1:
        writer1.write(image1)
    if ret2:
        writer2.write(image2)
    tock = time.time()
    print(f"writer.write took {tock-tick}")
    # count += 1
    # if count == 1:
    #     print("camera start")
    #cv2.imshow('Web1', image1)
    #cv2.imshow('Web2', image2)
    #raw_input("continue") # python2 :(
    plt.imshow(image1)
    plt.show()
    angle = get_angle(image1)
    print(f"angle is {angle}")
    #input("continue ")
    
cam1.release()
cam2.release()
writer1.release()
writer2.release()
cv2.destroyAllWindows()
print("camera stop")

import cv2
import sys
import numpy as np
from PIL import Image
import argparse
import os
import numpy as np



hMin = 0
sMin = 0
vMin = 50

hMax = 179  
sMax = 255
vMax = 255

def threshold(img, hMin = 0, sMin = 0, vMin = 50, 
    hMax = 179, sMax = 255 ,vMax = 255):

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])


    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    return output


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image to be threshed")
    #ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    #	help="type of preprocessing to be done")
    args = vars(ap.parse_args())
    img = cv2.imread(args["image"])

    threshold(img)

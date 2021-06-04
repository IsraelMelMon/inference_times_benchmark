"""
Preprocessing pipeline for tomato detection 
from the data set: delValle:primera, resaga.

Execute an example as:
 bb_clasico.py -i images/folder -o output/folder -c rgb

 creates a bounding box over the original RGB images and 
 writes them to the output folder


By: Israel Melendez Montoya for 
Rochin Industrias 
"""
import argparse
import os
import os.path
from os import path
import shutil

import cv2
import numpy as np

def draw_bounding_boxes(i, img, output_folder):
    #image = cv2.resize(image, (0,0), fx=1.5,fy=1.5)
    # read the image
    image = cv2.imread(img)

    image_copy=image.copy()

    hMin = 0
    sMin = 0
    vMin = 50
    hMax = 179
    sMax = 255 
    vMax = 255

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])


    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image_copy,image_copy, mask= mask)
    
    
    img_bw = 255*(cv2.cvtColor(output, 
        cv2.COLOR_BGR2GRAY) > 10).astype('uint8')

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    output_closed_image = output * mask

    # convert to RGB
    #image_2 = cv2.cvtColor(output_closed_image, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(output_closed_image.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    
    # create a binary thresholded image
    #_, binary = cv2.threshold(output_closed_image, 30, 255, cv2.THRESH_BINARY)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(gray, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # draw all contours
    contour_counter = 0
    # initiate lines
    boundRect= [None]*len(contours)
    #print(contours)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1000:
            boundRect[i] = cv2.boundingRect(contour)

            #image = cv2.drawContours(image, contour, 
                #-1, (0, 255, 0), 3)

            
            cv2.rectangle(image, (int(boundRect[i][0]), 
                int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]),
                int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
            
            contour_counter = contour_counter + 1
    
    #stop = time.time()

    """font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (300,200)
    fontScale              = 2
    fontColor              = (0,255,0)
    lineType               = 3

    cv2.putText(image, str(contour_counter), 
        topLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    """

    output_image = "bboxes_"+img
    cv2.imwrite(os.path.join(output_folder,output_image), image)

if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input image folder to count tomatoes")

    ap.add_argument("-o", "--output", required=False,
        help="path to output image folder to bounding boxed tomatoes")
    
    ap.add_argument("-c", "--color", required=False, type=str,
        help="Read RGB or grey scale images: rgb or grey")

    args = vars(ap.parse_args())

    print("[INFO] Reading files in folder...")

    imgs = os.listdir(args["input"])
    imgs = [img_name for img_name in imgs if "COL" in img_name ]
    #print(imgs)
    while ".DS_Store" in imgs: imgs.remove(".DS_Store")    
    
    
    if not os.path.exists("outputs"):
        try: 
            os.mkdir("outputs")

            print(("[INFO] Creating the output folder..."))
                  
            for i, img in enumerate(imgs):
                draw_bounding_boxes(i, img, output_folder = "outputs" )
            print("[INFO] Finished writing bounding boxes at /outputs" )
            

        except OSError:
            print(("[INFO] The directory /outputs could not "
             " be created"))
            
    else:
        print(("[INFO] The output directory already exists, "
        "overwriting"))

        shutil.rmtree("outputs")

        os.mkdir("outputs")
        
        for i, img in enumerate(imgs):
            draw_bounding_boxes(i, img, output_folder = "outputs" )
        print("[INFO] Finished writing bounding boxes at /outputs" )
            

    #print("[INFO] Drawing bounding boxes and writing as annotations...")
    #draw_bounding_boxes(filenames)

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
import time 

import cv2
import tensorflow as tf

import numpy as np



def initialize_model(model_file):

    interpreter = tf.lite.Interpreter(
    model_path=model_file, num_threads=8)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return width, height, interpreter, floating_model, input_details, output_details

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def draw_bounding_boxes(i, img, output_folder, input_folder, 
                            crop, inference, model_file, label_file):
    #image = cv2.resize(image, (0,0), fx=1.5,fy=1.5)
    # read the image
    image = cv2.imread(os.path.join(input_folder, img))

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

    # convert to grayscale
    gray = cv2.cvtColor(output_closed_image.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    
    # create a binary thresholded image

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(gray, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # draw all contours
    contour_counter = 0
    # initiate lines
    boundRect= [None]*len(contours)

    start_time = time.time()
    for i, contour in enumerate(contours):

        if cv2.contourArea(contour) > 1000:
            boundRect[i] = cv2.boundingRect(contour)
            x, y, w, h = cv2.boundingRect(contour)
            #image = cv2.drawContours(image, contour, 
                #-1, (0, 255, 0), 3)
            crop_img = image[y:y+h, x:x+w]
            output_image = str(contour_counter)+str(args["input"][:-1])+"_tomatoes_"+img
            crop_img = cv2.resize(crop_img, (0,0), fx=2.0, fy=2.0)

            if inference==1:

                (width, height, interpreter, floating_model, input_details, 
                    output_details) = initialize_model(model_file)

                crop_img = cv2.resize(crop_img, (224,224))
                # add N dim
                input_data = np.expand_dims(crop_img, axis=0)

                if floating_model:
                    input_data = (np.float32(input_data) ) / 255.0

                interpreter.set_tensor(input_details[0]["index"], input_data)

                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]["index"])
                results = np.squeeze(output_data)

                top_k = results.argsort()[:][::-1]
                labels = load_labels(args["label_file"])

                for i in top_k:
                    if floating_model:
                        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
                    else:
                        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

            if crop==1:
                cv2.imwrite(os.path.join(output_folder, output_image), crop_img)
            elif crop==-1:
                cv2.rectangle(image, (int(boundRect[i][0]), 
                    int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]),
                    int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
            else:
                pass

            contour_counter = contour_counter + 1
    
    stop_time = time.time()
    time_ms = (stop_time - start_time) * 1000
    print("time per picture (ms)", time_ms)

    if crop==-1:
        output_image = "bboxes_"+img
        cv2.imwrite(os.path.join(output_folder,output_image), image)

    return contour_counter

if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input image folder to count tomatoes")

    ap.add_argument("-o", "--output", required=False, default="outputs",
        help="path to output image folder to bounding boxed tomatoes")
    
    ap.add_argument("-n", "--number", required=False, default=100, type=int,
        help="number of images to process. default: 100")
    
    ap.add_argument("-c", "--color", required=False, type=int, default=1,
        help="Read RGB or grey scale images: rgb or grey")

    ap.add_argument("-cr", "--crop", required=False, type=int, default=0,
        help="Saves 2x resized crops of bounding boxes detected in output file, "
        "options: 0-doesn't save any imgs, 1- saves bb crops, "
        "-1 - saves full img with drawn rectangles")

    ap.add_argument("-inf", "--inference", required=False,  type=int, default=0,
        help="Returns inference over 2x resized crops with a tflite model")

    ap.add_argument("-m", "--model", required=False,  type=str, 
        default="models/lite_bi_rochin_2_tomate_grape_color.tflite",
        help="Tflite model file to use.")

    ap.add_argument("-l","--label_file", required = False, default="class_labels.txt",
        help="name of file containing labels")


    args = vars(ap.parse_args())

    print("[INFO] Reading files in folder...")

    imgs = os.listdir(args["input"])
    if args["color"]==1:
        imgs = [img_name for img_name in imgs if "COL" in img_name ]
    else:
        imgs = [img_name for img_name in imgs if "BN" in img_name ]

    # Limits the number of images to be processed
    if args["number"]:
        imgs = imgs[:int(args["number"])]

    while ".DS_Store" in imgs: imgs.remove(".DS_Store")    
    
    #if inference==1:
    #    width, height, interpreter, floating_model, input_details, output_details = initialize_model(model_file)



    if not os.path.exists("outputs"):
        try: 
            os.mkdir("outputs")

            print(("[INFO] Creating the output folder..."))
            total_boxes = []
            for i, img in enumerate(imgs):
                n_boxes = draw_bounding_boxes(i, img, output_folder = args["output"],
                    input_folder=args["input"], crop=args["crop"], 
                    inference=args["inference"], model_file=args["model"],
                    label_file=args["label_file"] )
                total_boxes.append(n_boxes)

            print("[INFO] Finished writing bounding boxes at /outputs" )
            print("[INFO] Total number of bounding boxes: ", sum(total_boxes))
            

        except OSError:
            print(("[INFO] The directory {} could not "
             " be created".format(args["output"])))
            
    else:
        print(("[INFO] The output directory already exists, "
        "overwriting"))
        total_boxes = []
        shutil.rmtree(args["output"])

        os.mkdir(args["output"])
        
        for i, img in enumerate(imgs):
            n_boxes = draw_bounding_boxes(i, img, output_folder = args["output"], 
                input_folder=args["input"], crop=args["crop"],inference=args["inference"],
                model_file=args["model"], label_file=args["label_file"] )

            total_boxes.append(n_boxes)
        print("[INFO] Finished writing bounding boxes at {}".format(args["output"]) )

        print("[INFO] Total number of bounding boxes: ", sum(total_boxes))
            

    #print("[INFO] Drawing bounding boxes and writing as annotations...")
    #draw_bounding_boxes(filenames)

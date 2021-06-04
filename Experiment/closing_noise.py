import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
#ap.add_argument("-p", "--preprocess", type=str, default="thresh",
#	help="type of preprocessing to be done")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])

img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10).astype('uint8')

se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

mask = np.dstack([mask, mask, mask]) / 255
out = img * mask


cv2.imshow('Output', cv2.resize(out,(0,0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("closed_"+args["image"], out)
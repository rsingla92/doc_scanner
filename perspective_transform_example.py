from doc_scanner import four_point_transform
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to the source image file")
ap.add_argument("-c", "--coords", help="CSV list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

warped = four_point_transform(image, pts)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

import numpy as np
import cv2


# From https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# and
# from https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

def order_points(pts):
    # Sort list of coordinates clockwise
    # beginning with the top left corner

    rect = np.zeros((4, 2), dtype="float32")

    # top left is the smallest sum
    # bottom right is the largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top right will have smallest difference
    # bottom left will have largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect

    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[0] - top_left[0]) ** 2))
    maxWidth = max(int(bottom_width), int(top_width))

    left_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    maxHeight = max(int(left_height), int(right_height))

    # determine where the points get transformed to.
    # take it from the camera view to the ideal screen view.

    dest_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dest_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

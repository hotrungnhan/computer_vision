import numpy as np
import cv2 as cv
filename = 'chessboard.png'
img = cv.imread(filename)  # import image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # generate gray scale image
# find Harris corners
gray = np.float32(gray)  # convert grey scale int to float32
dst = cv.cornerHarris(gray, 2, 3, 0.04)  # detect cornor harris  -> mask
# process mask the bright area brighter,  dark area darker
dst = cv.dilate(dst, None)
# filter bright pixel which greater than 0.01* brightest in dst
ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)  # convert mask -> int
ret, labels, stats, centroids = cv.connectedComponentsWithStats(
    dst)  # find centroids
# define the criteria to stop and refine the corners
# specify  iteration and accuracy  https://docs.opencv.org/4.5.2/d1/d5c/tutorial_py_kmeans_opencv.html
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

corners = cv.cornerSubPix(gray, np.float32(
    centroids), (5, 5), (-1, -1), criteria) #iteration to find sub-pixel accurate location of corners
 
# Now draw them
# generate centroit corner from harris detect
res = np.hstack((centroids, corners))
res = np.int0(res)  # res to signed int
img[res[:, 1], res[:, 0]] = [0, 0, 255]  # mark centroid
img[res[:, 3], res[:, 2]] = [0, 255, 0]  # mark corner from corner subpix
cv.imwrite('subpixel5.png', img)  # write output img

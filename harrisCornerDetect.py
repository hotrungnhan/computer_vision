import numpy as np
import cv2 as cv
filename = 'anotherchessboard.jpeg'  # declare filename
img = cv.imread(filename)  # read image
# resize image to be bigger due to it size so small
img = cv.resize(img, (780, 780), interpolation=cv.INTER_AREA)
# generate gray scale image to because 3 channel is not meaningful for connor dectect.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)  # conver image array from int to float32.

dst = cv.cornerHarris(gray, 2, 3, 0.04)  # detect connor by harris algrorithm

# result is dilated for marking the corners, not important
# dst = cv.dilate(dst, None)
# mask dst to img| Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.00001*dst.max()] = [0, 0, 255]
cv.imwrite('hasris_connor_detect'+filename, img)  # write output img
if cv.waitKey(0) & 0xff == 27:  # quit window .
    cv.destroyAllWindows()

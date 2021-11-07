import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
filename = "anotherchessboard.jpeg"
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 1, 255, -1)
# cv.imshow('dst', img)
cv.imwrite('shitomatic_connor_detect'+filename, img)  # write output img
# if cv.waitKey(0) & 0xff == 27:  # quit window .
#     cv.destroyAllWindows()

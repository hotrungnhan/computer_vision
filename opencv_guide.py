import cv2
import imutils
img = cv2.imread("./blackpink.jpg")


# get image dimension, dept
(h, w, d) = img.shape
print("width={}, height={}, depth={}".format(w, h, d))

# get invidual pixel
(B, G, R) = img[100, 50]
print("R={}, G={}, B={}".format(R, G, B))
cv2.imshow("Image", img)
# crop
roi = img[60:160, 320:420]
cv2.imshow("ROI", roi)
# resize image
resized = cv2.resize(img, (200, 200))
cv2.imshow("Fixed Resizing", resized)
# resize keep aspect ratio
resized = imutils.resize(img, width=300)
cv2.imshow("Imutils Resize", resized)
# smooth or blur
blurred = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow("Blurred", blurred)
#draw 
output = img.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

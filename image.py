import cv2
face_cascade_name = cv2.data.haarcascades + \
    'haarcascade_frontalface_alt2.xml'  # get default cascade model from cv2
face_classifier = cv2.CascadeClassifier()  # create cascade
# try load model -> fatal if getting false
if not face_classifier.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)

img = cv2.imread("./blackpink.jpg")  # read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # tranform image in to gray color
faces = face_classifier.detectMultiScale(
    gray, 1.1, 2)  # detect face in gray image
blurred = cv2.GaussianBlur(gray, (11, 11), 0)  # denoise
for index, (x, y, w, h) in enumerate(faces):  # draw bounding box for each face
    print("facedetected")
    cv2.rectangle(blurred, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(blurred, str(index), (x+w, y+h),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0))
cv2.imshow("img", blurred)  # render image to canvas
while True:  # hold canvas open
    k = cv2.waitKey(30)
    if k == 27:
        break

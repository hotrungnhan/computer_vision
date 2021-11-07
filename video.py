import cv2
from time import sleep
face_cascade_name = cv2.data.haarcascades + \
    'haarcascade_frontalface_alt2.xml'  # get default cascade model from cv2
face_classifier = cv2.CascadeClassifier()  # create cascade
# try load model -> fatal if getting false
if not face_classifier.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)
videoStream = cv2.VideoCapture(1)  # instantiate video capture
while True:
    _, img = videoStream.read()  # read each frame
    # tranform image in to gray color
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (11, 11), 0)  # denoise
    faces = face_classifier.detectMultiScale(
        img, 1.1, 2)  # detect face in gray image


    for (x, y, w, h) in faces:  # draw bounding box for each face
        print("facedetected")
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("img", img)  # render image to canvas

    k = cv2.waitKey(30)   # check for exist
    if k == 27:
        break
    sleep(0.0166666667)  # limit interval of detect

videoStream.release()  # release camera resource to be free

import cv2

cap = cv2.VideoCapture("http://169.254.166.3/mjpg/video.mjpg")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# tracker = cv2.TrackerMedianFlow_create()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey()

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
lowerbody_cascade = cv2.CascadeClassifier("haarcascade_lowerbody.xml")
upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
fullbody_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")


vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame)
    eyes = eye_cascade.detectMultiScale(gray_frame)
    smile = smile_cascade.detectMultiScale(gray_frame)
    lowerbody = lowerbody_cascade.detectMultiScale(gray_frame)
    upperbody = upperbody_cascade.detectMultiScale(gray_frame)
    fullbody = fullbody_cascade.detectMultiScale(gray_frame)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # for (x, y, w, h) in eyes:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # # for (x, y, w, h) in smile:
    # #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # for (x, y, w, h) in lowerbody:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (125, 30, 255), 2)
    # for (x, y, w, h) in upperbody:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 255, 140), 2)
    for (x, y, w, h) in fullbody:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (190, 200, 255), 2)

    cv2.imshow("Live Camera",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


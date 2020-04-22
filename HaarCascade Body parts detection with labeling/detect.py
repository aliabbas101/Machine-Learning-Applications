import cv2


def detect_eye(frame):
    eyes = eye_cascade.detectMultiScale(frame)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(roi_color, 'eye', (ex + 6, ey - 6), font, 0.5, (0, 
        255, 0), 1)




if __name__ == "__main__":
    cap = cv2.VideoCapture(0)


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,155,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, 'face', (x + 6, y - 6), font, 0.5, (0, 
            255, 0), 1)
            
            detect_eye(roi_gray)

        cv2.imshow('Haarcascade Detection',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()





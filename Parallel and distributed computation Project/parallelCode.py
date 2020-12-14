import numpy as np
import threading
import cv2

face_cascade = cv2.CascadeClassifier \
        (
        r"C:\Users\henri\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier \
    (r"C:\Users\henri\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier \
    (r"C:\Users\henri\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_smile.xml")
hand_cascade = cv2.CascadeClassifier \
    (r"C:\Users\henri\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\hand.xml")
hand_cascade1 = cv2.CascadeClassifier \
    (r"C:\Users\henri\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\Hand_haar_cascade.xml")

def eye(frame,roi_gray,roi_color):
    eye = eye_cascade.detectMultiScale(roi_gray, 1.2, 18)
    for (x_eye, y_eye, w_eye, h_eye) in eye:
        # drews a green rectangle if recognize eyes
        cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)
        cv2.putText(frame, "The eyes are focused on the camera", (00, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 60), False)
    return frame

def smile(frame,roi_gray,roi_color):
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    for (sx, sy, sw, sh) in smiles:
        # drews a red rectangle if recognize a smile
        cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        cv2.putText(frame, "Identify a smile", (00, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), False)
    return frame


def face(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # drews a blue rectangle if recognize a face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv2.putText(frame, "your face is front the camera", (00, x + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eye_process = threading.Thread(target=eye, args=(frame,roi_gray,roi_color))
        smile_process = threading.Thread(target=smile, args=(frame,roi_gray,roi_color))
        arr = [eye_process, smile_process]
        for p in arr:
            p.start()
        for p in arr:
            p.join()
    return frame




def hands(gray, frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)  # BLURRING IMAGE TO SMOOTHEN EDGES
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # BGR -> GRAY CONVERSION
    retval2, thresh1 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # THRESHOLDING IMAGE
    hand = hand_cascade1.detectMultiScale(thresh1, 1.3, 5)  # DETECTING HAND IN THE THRESHOLDE IMAGE
    mask = np.zeros(thresh1.shape, dtype="uint8")  # CREATING MASK
    for (x, y, w, h) in hand:  # MARKING THE DETECTED ROI
        cv2.rectangle(frame, (x, y), (x + w, y + h), (122, 122, 0), 2)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        cv2.putText(frame, "Identify a hands", (00, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 122, 0), False)
    # img2 = cv2.bitwise_and(thresh1, mask)
    # final = cv2.GaussianBlur(img2, (7, 7), 0)
    return frame


def detect(gray, frame):
    face_process = threading.Thread(target=face, args=(gray, frame))
    hand_process = threading.Thread(target=hands, args=(gray, frame))
    arr = [face_process, hand_process]
    for p in arr:
        p.start()
    for p in arr:
        p.join()
    return frame

cv2.destroyAllWindows()


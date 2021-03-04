import cv2
import numpy as np
import pytesseract
import threading
from imutils.video import VideoStream
from imutils import resize
from time import sleep
from datetime import datetime
from pathlib import Path

class Detector:
    def __init__(self):
        self.cap = VideoStream(0)
        self.cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)
        self.box = None
        self.frame = None
        self.cap.start()
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()


    def _alpr(self, img):
        number = '[]'
        plates = self.cascade.detectMultiScale(img ,scaleFactor=1.1, minNeighbors=3)
        if plates is not None:
            number = '[xxx]'
            x, y, w, h = sorted(plates, key=lambda x: x[3]*x[2], reverse=True)[0]
            image = img[y:y+h,x:x+w]
            image = resize(image, width=500)
            blur = cv2.bilateralFilter(image, 9, 75, 75)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray,
                                                 config =
                                                 f'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if text:
                number = f'[{text}]'
            cv2.imshow(number, gray)
            cv2.waitKey(0)
            return number


    def run(self):
        while True:
            #self.frame = self.cap.read()
            self.frame = cv2.imread(f"005.jpg")
            indx, scores, boxes = self.model.detect(self.frame, 0.8, 0.8)
            car_boxes = []
            for i in range(len(boxes)):
                if indx[i] in [2, 5, 6, 7]:
                    car_boxes.append(boxes[i])
            if car_boxes:
                print("car detected")
                self.box = sorted(car_boxes, key=lambda x: x[3]*x[2], reverse=True)[0]
                x, y, w, h = self.box
                detected_n = self._alpr(self.frame[y:y+h,x:x+w])
                timenow = datetime.now()
                str_p = f'./data/{timenow.strftime("%Y/%m/%d")}'
                path = Path(str_p)
                path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str_p + "/" + timenow.strftime("%H:%M:%S") +
                            detected_n + ".jpg", self.frame[y:y+h,x:x+w])
                self.detected = True
                print(f"car detected!")
            else:
                self.box = None
            sleep(5)


def test(det):
    while True:
        if cv2.waitKey(1) == 27:
            break
        frame = det.frame.copy()
        if det.box is not None:
            x, y, w, h = det.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 255, 0], 2)
        cv2.imshow("frame", frame)


if __name__ == "__main__":
    det = Detector()
    try:
        while True:
            sleep(60)
            pass
    except KeyboardInterrupt:
        print("kb interrupt")
        exit()


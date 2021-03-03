import cv2
import threading
from time import sleep
from datetime import datetime
from anpr.ocr import PlateDetector

class Detector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)
        self.pd = PlateDetector()
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            sleep(5)
            _, frame = self.cap.read()
            indx, scores, boxes = self.model.detect(frame, 0.8, 0.8)
            detected_n = ""
            for i in range(len(boxes)):
                if indx[i] in [2, 5, 6, 7]:
                    detected_n = "[xxx]"
            if detected_n:
                cv2.imwrite(datetime.now().strftime("%d:%m:%Y %H:%M:%S") +
                           detected_n + ".jpg", frame)
                print(f"car detected! {indx[i]}")

if __name__ == "__main__":
    det = Detector()
    try:
        while True:
            sleep(0.1)
            pass
    except KeyboardInterrupt:
        print("kb interrupt")


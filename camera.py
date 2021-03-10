import cv2
import numpy as np
import os
import pymongo
import random
from PIL import Image


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 640)
        self.video.set(4, 480)
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # width = int(self.video .get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        # height = int(self.video .get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 第三个参数则是镜头快慢的，10为正常，小于10为慢镜头
        # self.out = cv2.VideoWriter('./output2.avi', fourcc,10,(width,height))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        mongodbs = pymongo.MongoClient('127.0.0.1', 27017)
        self.face = mongodbs.face
        self.path = "dataset"

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # frame = cv2.flip(image, 1)
        # a = self.out.write(frame)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    # 图片采集
    def data_set(self, face_name):
        count = 0
        face_id = random.randint(0, 100000)
        while (True):
            ret, img = self.video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite(self.path + "/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break
        self.face.face.save({"face_name": face_name, "face_id": face_id})
        self.training()

    # 图片训练
    def training(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        imagePaths = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # 认证函数
    def recognition(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        minW = 0.1 * self.video.get(3)
        minH = 0.1 * self.video.get(4)
        while True:
            id = 'unknown'
            ret, img = self.video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    face = self.face.face.find_one({"face_id": id})
                    id = id
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
            if id != "unknown":
                break
        return face['face_name']

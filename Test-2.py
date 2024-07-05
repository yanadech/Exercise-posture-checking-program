import cv2
import numpy as np
import time
import Test as pm
import pygame

class Cameraplank:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = pm.poseDetector()
        self.is_planking = False
        self.start_time = 0
        self.elapsed_time = 0
        self.back = False

        # ตั้งค่าเสียงเตือน
        pygame.mixer.init()
        self.correct_sound = "static/correct.mp3"
        self.lower_sound = "static/Loweryourback.mp3"
        self.raise_sound = "static/Raiseyourback.mp3"
        self.toward_sound = "static/armtoward.mp3"
        self.away_sound = "static/armaway.mp3"

    def __del__(self):
        self.cap.release()

    def log_count(self, elapsed_time):
        with open("plank_log.txt", "a") as f:
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} - Count: {int(elapsed_time)}\n")

    def countdown_timer(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            remaining_time = int(end_time - time.time())
            print(f"Time remaining: {remaining_time} seconds", end="\r")
            time.sleep(1)

    def test_model(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = self.detector.preprocess_frame(img)
            img = self.detector.findPose(img)
            lmList = self.detector.findPosition(img)
            lmList = self.detector.postprocess_results(lmList)
            # แสดงผล
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camplank = Cameraplank()
    camplank.test_model()

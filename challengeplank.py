import cv2
import numpy as np
import time
import PosEstimationModule as pm
import pygame

class Challengeplank:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = pm.poseDetector()
        self.is_planking = False
        self.start_time = 0
        self.elapsed_time = 60
        self.incorrect_posture = False
        self.correct_posture_once = False
        self.current_sound = None 
        self.counter_time = 0

        # ตั้งค่าเสียงเตือน
        pygame.mixer.init()
        self.correct_sound = "static/correct.mp3"
        self.lower_sound = "static/Loweryourback.mp3"
        self.raise_sound = "static/Raiseyourback.mp3"
        self.toward_sound = "static/armtoward.mp3"
        self.away_sound = "static/armaway.mp3"
        self.count_sound = "static/count.mp3" 
    
    def __del__(self):
        self.cap.release()

    def play_sound(self, sound_file):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            self.current_sound = sound_file

    def get_frame(self):
        success, img = self.cap.read()
        if not success:
            return None

        img = cv2.resize(img, (1280, 720))
        img = self.detector.findPose(img, False)
        lmList = self.detector.getPosition(img)

        if len(lmList) != 0:
            angle_back_left = self.detector.findAngle(img, 11, 23, 27)
            angle_back_right = self.detector.findAngle(img, 12, 24, 28)
            angle1 = (angle_back_left + angle_back_right) / 2

            angle_arm_left = self.detector.findAngle(img, 11, 13, 15)
            angle_arm_right = self.detector.findAngle(img, 12, 14, 16)
            angle2 = (angle_arm_left + angle_arm_right) / 2

            hand_thresh = lmList[24][2]
            hands_on_ground = False

            if len(lmList) > 16:
                left_hand_y = lmList[15][2]
                right_hand_y = lmList[16][2]

                if left_hand_y > hand_thresh or right_hand_y > hand_thresh:
                    hand_status = "Hands on ground"
                    hands_on_ground = True
                else:
                    hand_status = "Hands in air"
                    hands_on_ground = False

                cv2.putText(img, hand_status, (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

            back_status = True
            if angle1 <= 160:
                self.back = "Back : Too Low"
                back_status = False
                if self.current_sound != self.raise_sound:
                    self.play_sound(self.raise_sound)
            elif angle1 >= 190:
                self.back = "Back : Too High"
                back_status = False
                if self.current_sound != self.lower_sound:
                    self.play_sound(self.lower_sound)
            else:
                self.back = "Back : Correct"
                back_status = True
                if self.current_sound != self.correct_sound:
                    self.play_sound(self.correct_sound)

            cv2.putText(img, self.back, (50, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

            if angle2 <= 230:
                arm = "Arm : Too Far"
                arm_status = False
                if self.current_sound != self.toward_sound:
                    self.play_sound(self.toward_sound)
            elif angle2 >= 260:
                arm = "Arm : Too Close"
                arm_status = False
                if self.current_sound != self.away_sound:
                    self.play_sound(self.away_sound)
            else:
                arm = "Arm : Correct"
                arm_status = True
                if self.current_sound != self.correct_sound:
                    self.play_sound(self.correct_sound)

            correct_posture = back_status and hands_on_ground and arm_status

            if correct_posture:
                if not self.correct_posture_once:
                    self.correct_posture_once = True
                    self.start_time = time.time()
                    if self.current_sound != self.count_sound:
                        self.play_sound(self.count_sound)
                self.elapsed_time = time.time() - self.start_time
            elif self.correct_posture_once:
                self.incorrect_posture = True

            cv2.putText(img, arm, (50, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
            cv2.putText(img, f'Time: {int(self.elapsed_time)} s', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img

    def start_challenge(self, duration=60):
        self.start_time = time.time()
        while True:
            elapsed_time = time.time() - self.start_time
            self.counter_time = time.time() - self.start_time
            if elapsed_time > duration:
                break
            
            frame = self.get_frame()
            if frame is None:
                continue
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg.tobytes()
            
        # เมื่อครบเวลา
        if self.elapsed_time <= 0:
            cv2.putText(frame, 'Success', (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f'Plank: {int(self.counter_time)} s', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Fail', (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, f'Plank: {int(self.counter_time)} s', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield jpeg.tobytes()

        self.cap.release()


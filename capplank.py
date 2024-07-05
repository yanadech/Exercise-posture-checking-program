import cv2
import numpy as np
import time
import PosEstimationModule as pm
import pygame

class Cameraplank:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = pm.poseDetector()
        self.is_planking = False
        self.start_time = 0
        self.elapsed_time = 0
        self.back = False
        self.current_sound = None 
        self.arm = False
        
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

            back_status = True
            if angle1 >= 190:
                self.back = "Back : Too High"
                back_status = False
                if self.current_sound != self.lower_sound:
                    self.play_sound(self.lower_sound)
            elif angle1 <= 160:
                self.back = "Back : Too Low"
                back_status = False
                if self.current_sound != self.raise_sound:
                    self.play_sound(self.raise_sound)
            else:
                self.back = "Back : Correct"
                back_status = True
                if self.current_sound != self.correct_sound:
                    self.play_sound(self.correct_sound)

            cv2.putText(img, str(self.back), (50, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

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

                cv2.putText(img, hand_status, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

            arm_status = True

            if angle2 <= 230:
                self.arm = "Arm : Too Far"
                arm_status = False
                if self.current_sound != self.toward_sound:
                    self.play_sound(self.toward_sound)
            elif angle2 >= 260:
                self.arm = "Arm : Too close"
                arm_status = False
                if self.current_sound != self.away_sound:
                   self.play_sound(self.away_sound)
            else:
                self.arm = "Arm : Correct"
                arm_status = True
                if self.current_sound != self.correct_sound:
                    self.play_sound(self.correct_sound)

            cv2.putText(img, str(self.arm), (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

            if back_status and hands_on_ground and arm_status:
                if not self.is_planking:
                    self.is_planking = True
                    self.start_time = time.time()
                    if self.current_sound != self.count_sound:
                        self.play_sound(self.count_sound)
                self.elapsed_time = time.time() - self.start_time
            else:
                if self.is_planking:
                    self.is_planking = False

            cv2.putText(img, f'Time: {int(self.elapsed_time)} s', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img

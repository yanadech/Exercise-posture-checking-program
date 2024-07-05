import cv2
import numpy as np
import time
import PosEstimationModule as pm
import pygame

class Cameralungs:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # วิดีโอสำหรับท่า Lunge
        self.detector = pm.poseDetector()
        self.count = 0
        self.dir = 0
        self.back_status = False
        self.current_sound = None 
        self.start_check_complete = False
        
        # ตั้งค่าเสียงเตือน
        pygame.mixer.init()
        self.correct_sound = "static/correct.mp3"
        self.lower_sound = "static/Loweryourback.mp3"    
        self.raise_sound = "static/Raiseyourback.mp3"  
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

        back = None 
        
        if len(lmList) != 0:
            # ตรวจสอบมุมข้อเข่า
            angle_knee_left = self.detector.findAngle(img, 23, 25, 27)  # มุมเข่าหน้าซ้าย
            angle_knee_right = self.detector.findAngle(img, 24, 26, 28) # มุมเข่าหน้าขวา

            avg_knee = (angle_knee_right + angle_knee_left) / 2
            per1 = np.interp(avg_knee, (186, 230), (0, 100))

            # มุมหลัง (ไหล่, สะโพก, เข่า)
            back_angle_left = self.detector.findAngle(img, 11, 23, 25)   # ไหล่ซ้าย, สะโพกซ้าย, เข่าซ้าย
            back_angle_right = self.detector.findAngle(img, 12, 24, 26)  # ไหล่ขวา, สะโพกขวา, เข่าขวา

            avg_back = (back_angle_right + back_angle_left) / 2

            # ตรวจสอบว่ามือสัมผัสพื้นหรือไม่
            foot_thresh = 10  # ค่าเกณฑ์ความสูงของพื้นในหน่วยพิกเซล
            feet_on_ground = False  # ตัวแปรเพื่อตรวจสอบสถานะเท้า

            if len(lmList) > 16:  # ตรวจสอบว่ามีข้อมูลตำแหน่งของ landmark 15 และ 16 หรือไม่
                left_foot_y = lmList[27][2]  # y-coordinate ของเท้าซ้าย
                right_foot_y = lmList[28][2]  # y-coordinate ของเท้าขวา
                heel_left_y = lmList[29][2]  # y-coordinate ของส้นเท้าซ้าย
                heel_right_y = lmList[30][2]  # y-coordinate ของส้นเท้าขวา

                if (left_foot_y > foot_thresh and heel_left_y > foot_thresh) or (right_foot_y > foot_thresh and heel_right_y > foot_thresh):
                    foot_status = "Feet on ground"
                    feet_on_ground = True
                else:
                    foot_status = "Feet in air"
                    feet_on_ground = False

                # แสดงสถานะเท้า
                cv2.putText(img, foot_status, (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

            # ตรวจสอบหลัง
            if avg_back <= 160:
                back = "Back : Too Low"
                self.back_status = False
                if self.current_sound != self.raise_sound:
                    self.play_sound(self.raise_sound)
            elif avg_back >= 190:
                back = "Back : Too High"
                self.back_status = False
                if self.current_sound != self.lower_sound:
                    self.play_sound(self.lower_sound)
            else:
                back = "Back : Correct"
                self.back_status = True

            if not self.start_check_complete and feet_on_ground:
                if self.back_status :
                    self.start_check_complete = True
                    self.play_sound(self.correct_sound)
                

            # วาดแถบสถานะ
            color = (255, 0, 255)
            if per1 == 100:
                color = (0, 255, 0)
                if self.dir == 0 and self.back_status and feet_on_ground:
                    self.count += 0.5
                    self.dir = 1
            if per1 == 0:
                color = (0, 255, 0)
                if self.dir == 1 and self.back_status and feet_on_ground:
                    self.count += 0.5
                    self.dir = 0
                    if self.current_sound != self.count_sound:
                        self.play_sound(self.count_sound)

            # วาดจำนวนการทำ Lunge
            cv2.rectangle(img, (0, 550), (300, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(self.count)), (45, 700), cv2.FONT_HERSHEY_PLAIN, 13, (255, 0, 0), 15)

            # บาร์นับจำนวน
            if back == "Back : Correct" and feet_on_ground:
                color = (0, 255, 0)
            else: 
                color = (0, 0, 255)

            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(np.interp(avg_knee, (186, 215), (650, 100)))), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per1)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # แสดงสถานะหลัง
        cv2.putText(img, str(back), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        return img

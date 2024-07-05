import cv2
import numpy as np
import time
import PosEstimationModule as pm
import pygame

class Challengesquat:
    def __init__(self):
        # เริ่มต้น pose detector
        self.detector = pm.poseDetector()
        self.count = 0
        self.dir = 0
        self.back = None
        self.back_status = None
        self.cap = cv2.VideoCapture(0)
        self.target_squat = 20 # จำนวนที่ต้องการ
        self.over_squat = 0
        self.remaining_squat = self.target_squat
        self.current_sound = None 
        self.start_check_complete = False
        self.first_correct_completed = False

        # เริ่มต้น pygame สำหรับเสียงเตือน
        pygame.mixer.init()
        self.correct_sound = "static/correct.mp3"
        self.lower_sound = "static/Loweryourback.mp3"    
        self.raise_sound = "static/Raiseyourback.mp3"  
        self.count_sound = "static/count.mp3" 
        self.incorrect_sound = "static/incorrect.mp3"
    
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
            return None, self.count, self.remaining_squat

        img = cv2.resize(img, (1280, 720))
        img = self.detector.findPose(img, False)
        lmList = self.detector.getPosition(img)

        if len(lmList) != 0:
            angle_knee_left = self.detector.findAngle(img, 23, 25, 27)
            angle_knee_right = self.detector.findAngle(img, 24, 26, 28)

            avg_knee_angle = (angle_knee_left + angle_knee_right) / 2
            per1 = np.interp(avg_knee_angle, (180, 230), (0, 100))
            bar = np.interp(avg_knee_angle, (180, 230), (650, 100))

            back_angle_left = self.detector.findAngle(img, 11, 23, 25)
            back_angle_right = self.detector.findAngle(img, 12, 24, 26)
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
                self.back = "Back : Too Low"
                if self.current_sound != self.raise_sound:
                    self.play_sound(self.raise_sound)
                self.back_status = False
            elif avg_back >= 190:
                self.back = "Back : Too High"
                if self.current_sound != self.lower_sound:
                    self.play_sound(self.lower_sound)
                self.back_status = False
            else:
                self.back = "Back : Correct"
                self.back_status = True

            if not self.start_check_complete:
                if self.back_status and feet_on_ground:
                    self.start_check_complete = True
                    self.first_correct_completed = True
                    self.play_sound(self.correct_sound)
                
        if self.first_correct_completed:
            if per1 == 100 and self.back_status and feet_on_ground:
                if self.dir == 0:
                    self.count += 0.5
                    self.dir = 1
                    self.remaining_squat -= 0.5
            if per1 == 0 and self.back_status and feet_on_ground:
                if self.dir == 1:
                    self.count += 0.5
                    self.dir = 0
                    self.remaining_squat -= 0.5
                    if self.current_sound != self.count_sound:
                        self.play_sound(self.count_sound)

            if not (self.back_status and feet_on_ground):
                # แสดงข้อความหรือทำสิ่งที่ต้องการเมื่อพบการทำผิด
                self.play_sound(self.incorrect_sound)
                cv2.putText(img, 'Fail', (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.putText(img, f'Squats: {int(self.count)}', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                self.cap.release()
                return img
        
        self.over_squat = self.count - self.target_squat

        cv2.rectangle(img, (0, 550), (300, 720), (0, 255, 0), cv2.FILLED)
        if self.remaining_squat >= 0.5:
            cv2.putText(img, str(int(self.remaining_squat)), (45, 700), cv2.FONT_HERSHEY_PLAIN, 13, (255, 0, 0), 15)
        else:
            cv2.putText(img, str(int(self.over_squat)), (45, 700), cv2.FONT_HERSHEY_PLAIN, 13, (255, 0, 0), 15)
        # บาร์นับจำนวน
        if self.back == "Back : Correct":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per1)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.putText(img, str(self.back), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        return img  
      
    def start_challenge(self, duration=60):
        self.start_time = time.time()
        # ตรวจสอบท่าทางก่อนเริ่มจับเวลา
        while not self.start_check_complete:
            frame = self.get_frame()
            if frame is None:
                continue

            # แสดงข้อความให้ผู้ใช้รู้ว่ากำลังตรวจสอบท่าทาง
            cv2.putText(frame, 'Checking posture...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield jpeg.tobytes()

        # ตั้งค่าเวลาเริ่มต้นใหม่ทันทีหลังจากที่ท่าทางถูกต้อง
        self.start_time = time.time()

        while True:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > duration:
                break
            
            frame = self.get_frame()
            if frame is None:
                continue
            
            # เพิ่มข้อความนับเวลาถอยหลังในเฟรม
            remaining_time = duration - int(elapsed_time)
            cv2.putText(frame, f'Time left: {remaining_time}s', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg.tobytes()
        # เมื่อครบเวลา
        if self.remaining_squat <= 0:
            cv2.putText(frame, 'Success', (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f'Squats: {int(self.count)}', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Fail', (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, f'Squats: {int(self.count)}', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield jpeg.tobytes()

        # เมื่อนับเวลาถอยหลังเสร็จสิ้น
        self.cap.release()
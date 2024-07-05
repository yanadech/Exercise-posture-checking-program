import cv2
import mediapipe as mp
import time
import math
import numpy as np
from filterpy.kalman import KalmanFilter

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.8, trackCon=0.8, filter_window=5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.filter_window = filter_window
        self.pTime = 0
        self.angle_filter = {}

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.], [0.]])
        self.kf.F = np.array([[1., 1.], [0., 1.]])
        self.kf.H = np.array([[1., 0.]])
        self.kf.P *= 1000.
        self.kf.R = 5
        self.kf.Q = 0.1

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def applyKalmanFilter(self, id, cx, cy):
        if id not in self.angle_filter:
            self.angle_filter[id] = KalmanFilter(dim_x=4, dim_z=2)
            self.angle_filter[id].x = np.array([cx, cy, 0, 0])
            self.angle_filter[id].F = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
            self.angle_filter[id].H = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]])
            self.angle_filter[id].P *= 1000.
            self.angle_filter[id].R = 5
            self.angle_filter[id].Q = 0.1

        z = np.array([cx, cy])
        self.angle_filter[id].predict()
        self.angle_filter[id].update(z)
        return int(self.angle_filter[id].x[0]), int(self.angle_filter[id].x[1])

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cx, cy = self.applyKalmanFilter(id, cx, cy)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def filter_angle(self, angle):
        self.kf.predict()
        self.kf.update(angle)
        return self.kf.x[0, 0]

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        angle = self.filter_angle(angle)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 1)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 1)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return angle

    def giveFeedback(self, img, angle, threshold=10):
        if angle < threshold:
            cv2.putText(img, "Keep it up!", (50, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        else:
            cv2.putText(img, "Adjust your posture!", (50, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    def isValidPose(self):
        if len(self.lmList) < 33:
            return False

        required_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        for key, id in required_landmarks.items():
            x, y = self.lmList[id][1], self.lmList[id][2]
            if x <= 0 or y <= 0:
                return False

        shoulder_width = abs(self.lmList[required_landmarks['left_shoulder']][1] - self.lmList[required_landmarks['right_shoulder']][1])
        hip_width = abs(self.lmList[required_landmarks['left_hip']][1] - self.lmList[required_landmarks['right_hip']][1])
        if shoulder_width < 50 or hip_width < 50:
            return False

        # Check the relative positions of shoulders, hips, knees, and ankles
        if self.lmList[required_landmarks['left_shoulder']][2] > self.lmList[required_landmarks['left_hip']][2]:
            return False
        if self.lmList[required_landmarks['right_shoulder']][2] > self.lmList[required_landmarks['right_hip']][2]:
            return False
        if self.lmList[required_landmarks['left_hip']][2] > self.lmList[required_landmarks['left_knee']][2]:
            return False
        if self.lmList[required_landmarks['right_hip']][2] > self.lmList[required_landmarks['right_knee']][2]:
            return False

        return True

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        if detector.isValidPose():
            detector.showFps(img)
        else:
            cv2.putText(img, "Invalid Pose", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()

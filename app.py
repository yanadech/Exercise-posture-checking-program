from flask import Flask, render_template, Response
import cv2
from cappushup import Camerapushup
from capplank import Cameraplank
from capsquat import Camerasquat
from caplungs import Cameralungs
from challengelungs import Challengelungs
from challengeplank import Challengeplank
from challengepushup import Challengepushup
from challengesquat import Challengesquat
import numpy as np

app = Flask(__name__)

# Route สำหรับแต่ละท่าออกกำลังกาย
@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/pushup")
def pushup():
    return render_template("pushup.html")

@app.route("/squat")
def squat():
    return render_template("squat.html")

@app.route("/lung")
def lung():
    return render_template("lung.html")

@app.route("/plank")
def plank():
    return render_template("plank.html")

# Route สำหรับกล้องในแต่ละท่า
@app.route("/pushupcam")
def pushupcam():
    return render_template("pushupcam.html")

@app.route("/squatcam")
def squatcam():
    return render_template("squatcam.html")

@app.route("/lungcam")
def lungcam():
    return render_template("lungcam.html")

@app.route("/plankcam")
def plankcam():
    return render_template("plankcam.html")

# ฟังก์ชันการสร้าง frame สำหรับแต่ละท่า
def gen(camera, challenge=False):
    if challenge:
        for frame in camera.start_challenge():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    else:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            if isinstance(frame, tuple):
                frame = frame[0]
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            jpeg_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')

@app.route("/video_pushup")
def video_pushup():
    return Response(gen(Camerapushup()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_squat")
def video_squat():
    return Response(gen(Camerasquat()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_lungs")
def video_lungs():
    return Response(gen(Cameralungs()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_plank")
def video_plank():
    return Response(gen(Cameraplank()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/challengepushup")
def challangepushup():
    return render_template("challengepushup.html")

@app.route("/challengesquat")
def challangesquat():
    return render_template("challengesquat.html")

@app.route("/challengeplank")
def challangeplank():
    return render_template("challengeplank.html")

@app.route("/challengelungs")
def challangelungs():
    return render_template("challengelungs.html")

@app.route("/video_challengepushup")
def video_challengepushup():
    return Response(gen(Challengepushup(), challenge=True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_challengesquat")
def video_challengesquat():
    return Response(gen(Challengesquat(), challenge=True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_challengelungs")
def video_challengelungs():
    return Response(gen(Challengelungs(), challenge=True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_challengeplank")
def video_challengeplank():
    return Response(gen(Challengeplank(), challenge=True), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

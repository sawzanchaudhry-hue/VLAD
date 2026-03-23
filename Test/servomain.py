#!/usr/bin/env python3

import pigpio
import numpy as np
import depthai as dai
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import pickle
import cv2
import time
import sys
import tty
import termios
import threading

from rplidar import RPLidar
from motor import Motor
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import RPLidarA1
from ipad import ZoomWebInterface
from faced import FaceDetectionThread
from face_servo import ServoTrackingThread

MAP_SIZE_PIXELS = 1000
MAP_SIZE_METERS = 10

# Start SLAM
slam = RMHC_SLAM(RPLidarA1(), MAP_SIZE_PIXELS, MAP_SIZE_METERS)

CORD_SAVE_FILE = "CORD_SAVE_FILE.pkl"
SAVED_X = 0
SAVED_Y = 0
SAVED_THETA = 0

if os.path.exists(CORD_SAVE_FILE):
    with open(CORD_SAVE_FILE, "rb") as f:
        state = pickle.load(f)
        SAVED_X = float(state["x"])
        SAVED_Y = float(state["y"])
        SAVED_THETA = float(state["theta"])
        print(f"Got: {SAVED_THETA}")

mapbytes = bytearray(500 * 500)

holder = 0
scanOut = 0
lock = threading.Lock()

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Failed to connect to pigpio daemon")

# Motor Object
m = Motor(pi)

# Zoom object
zoomUI = ZoomWebInterface(
    meeting_url="https://torontomu.zoom.us/j/8205042429?pwd=QnlKUXF5RHJTN3ZzNVpUaEtDdTNsQT09",
    host="0.0.0.0",
    port=5000
)

# Face + Servo Threads
faceThread = None
servoThread = None

# Lidar Object
PORT = "/dev/ttyUSB0"
lidar = RPLidar(PORT)


def runLidar():
    global holder
    global scanOut
    motor1EncoderCount = 0
    lidar.start_motor()
    lidar.clean_input()
    time.sleep(5)

    while True:
        for scan in lidar.iter_scans(max_buf_meas=5000):
            distances = [0] * 360
            scanOut = scan
            for data in scanOut:
                dataTemp = int(data[1])
                if 190 <= dataTemp < 240:
                    if distances[dataTemp] == 0 or data[2] < distances[dataTemp]:
                        distances[dataTemp] = data[2]
            slam.update(distances)


def runLidarOutput():
    global SAVED_X
    global SAVED_Y
    global SAVED_THETA

    prevX = 5000.0
    prevY = 5000.0
    prevTheta = 0

    while True:
        x, y, theta = slam.getpos()
        SAVED_X = SAVED_X + float(prevX - x)
        SAVED_Y = SAVED_Y + float(prevY - y)
        SAVED_THETA = SAVED_THETA + float(prevTheta - theta)

        print(f"\rx={SAVED_X}, y={SAVED_Y}, angle={SAVED_THETA}", end="", flush=True)

        prevX = x
        prevY = y
        prevTheta = theta

        if scanOut != 0:
            for data in scanOut:
                if (data[1] >= 345) or (data[1] <= 15):
                    if data[2] <= 450:
                        x = 2 + 2

        time.sleep(1)


def get_char():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


while True:
    c = get_char().upper()

    if c == "A":
        print("STARTED MOTORS")
        m.RUN_MOTORS = True

    elif c == "W":
        print("FORWARDS")
        m.motorRunQueue.put(("F", 1))

    elif c == "S":
        print("BACKWARDS")
        m.motorRunQueue.put(("B", 1))

    elif c == "Q":
        print("RIGHT SPIN")
        m.motorRightSpin(0)

    elif c == "E":
        print("LEFT SPIN")
        m.motorLeftSpin(0)

    elif c == "D":
        print("STOP MOVE")
        m.motorStop(0)

    elif c == "L":
        print("STARTED LIDAR")
        lidarThread = threading.Thread(target=runLidar, daemon=True)
        lidarThread.start()

    elif c == "K":
        print("Start Lidar Outputter Thread")
        lidarOutputThread = threading.Thread(target=runLidarOutput, daemon=True)
        lidarOutputThread.start()

    elif c == "P":
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            videoQueue = cam.requestOutput((640, 400)).createOutputQueue()
            pipeline.start()

            while pipeline.isRunning():
                videoIn = videoQueue.get()
                assert isinstance(videoIn, dai.ImgFrame)
                cv2.imshow("video", videoIn.getCvFrame())
                if cv2.waitKey(1) == ord("q"):
                    break

    elif c == "F":
        print("START FACE DETECTION + SERVO TRACKING")

        if faceThread is None or not faceThread.is_alive():
            faceThread = FaceDetectionThread(show_window=False)
            faceThread.start()
            print("Face detection thread started")
        else:
            print("Face detection already running")

        if servoThread is None or not servoThread.is_alive():
            servoThread = ServoTrackingThread(faceThread)
            servoThread.start()
            print("Servo tracking thread started")
        else:
            print("Servo tracking already running")

    elif c == "G":
        if faceThread is not None:
            result = faceThread.get_result()
            print("FACE SEEN:", result["face_seen"], "BBOX:", result["bbox"])
        else:
            print("Face thread not started")

    elif c == "H":
        print("STOP FACE DETECTION + SERVO TRACKING")

        if servoThread is not None:
            servoThread.stop()
            servoThread = None

        if faceThread is not None:
            faceThread.stop()
            faceThread.join(timeout=2)
            faceThread = None

    elif c == "Z":
        print("ZOOM INTERFACE")
        zoomUI.start()

    elif c == "J":
        print("STOPPED MOTORS")
        m.RUN_MOTORS = False

        if servoThread is not None:
            servoThread.stop()
            servoThread = None

        if faceThread is not None:
            faceThread.stop()
            faceThread.join(timeout=2)
            faceThread = None

        with open(CORD_SAVE_FILE, "wb") as f:
            pickle.dump(
                {
                    "x": str(float(SAVED_X)),
                    "y": str(float(SAVED_Y)),
                    "theta": str(float(SAVED_THETA))
                },
                f
            )

        pi.set_PWM_dutycycle(m.MOTOR1_GPIO, 0)
        pi.set_PWM_dutycycle(m.MOTOR2_GPIO, 0)

        try:
            lidar.stop()
            lidar.disconnect()
        except Exception:
            pass

        pi.stop()
        break

    elif c == "R":
        print("RESETING CORDS")
        with open(CORD_SAVE_FILE, "wb") as f:
            pickle.dump(
                {
                    "x": str(int(5000.0)),
                    "y": str(int(5000.0)),
                    "theta": str(int(0))
                },
                f
            )
        break

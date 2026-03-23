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
SAVED_X = 0.0
SAVED_Y = 0.0
SAVED_THETA = 0.0

if os.path.exists(CORD_SAVE_FILE):
    with open(CORD_SAVE_FILE, "rb") as f:
        state = pickle.load(f)
        SAVED_X = float(state["x"])
        SAVED_Y = float(state["y"])
        SAVED_THETA = float(state["theta"])
        print(f"Loaded saved theta: {SAVED_THETA}")

mapbytes = bytearray(500 * 500)

holder = 0
scanOut = 0
lock = threading.Lock()

# ================= STATE SYSTEM =================
current_state = "waiting"
target_room = None
status_message = "Waiting for nurse command"
patient_verified = False
zoom_ready = False

lidar_thread_started = False
lidar_output_thread_started = False
nav_thread_started = False

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Failed to connect to pigpio daemon")

# Motor Object
m = Motor(pi)

# Face + Servo Threads
faceThread = None
servoThread = None

# Lidar Object
PORT = "/dev/ttyUSB0"
lidar = RPLidar(PORT)


def get_ui_state():
    with lock:
        return {
            "current_state": current_state,
            "status_message": status_message,
            "target_room": target_room,
            "patient_verified": patient_verified,
            "zoom_ready": zoom_ready
        }


def handle_room_command(command):
    global current_state, target_room, status_message
    global patient_verified, zoom_ready
    global faceThread, servoThread

    with lock:
        patient_verified = False
        zoom_ready = False

        if command == "room1":
            target_room = "room1"
            current_state = "navigating"
            status_message = "Navigating to Room 1"

        elif command == "room2":
            target_room = "room2"
            current_state = "navigating"
            status_message = "Navigating to Room 2"

        elif command == "nurse_station":
            target_room = "nurse_station"
            current_state = "navigating"
            status_message = "Returning to Nurse Station"

        elif command == "start_zoom":
            if current_state == "ready_for_zoom":
                current_state = "zoom_active"
                status_message = "Zoom session active"


# Zoom object
zoomUI = ZoomWebInterface(
    meeting_url="https://torontomu.zoom.us/j/8205042429?pwd=QnlKUXF5RHJTN3ZzNVpUaEtDdTNsQT09",
    host="0.0.0.0",
    port=5000,
    command_callback=handle_room_command,
    state_callback=get_ui_state
)


def runLidar():
    global holder
    global scanOut

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
    global scanOut

    prevX = 5000.0
    prevY = 5000.0
    prevTheta = 0.0

    while True:
        x, y, theta = slam.getpos()

        with lock:
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
                        pass

        time.sleep(1)


def start_face_and_servo():
    global faceThread, servoThread

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


def stop_face_and_servo():
    global faceThread, servoThread

    if servoThread is not None:
        try:
            servoThread.stop()
        except Exception:
            pass
        servoThread = None

    if faceThread is not None:
        try:
            faceThread.stop()
            faceThread.join(timeout=2)
        except Exception:
            pass
        faceThread = None


def navigation_worker():
    global current_state, status_message, zoom_ready, patient_verified
    global target_room

    while True:
        with lock:
            state = current_state
            room = target_room

        if state == "navigating":
            print(f"\nNavigation worker: moving toward {room}")

            # Placeholder demo movement logic
            # Replace later with real navigation / arrival detection
            m.RUN_MOTORS = True
            m.motorRunQueue.put(("F", 0.2))
            time.sleep(2)
            m.motorStop(0)

            with lock:
                current_state = "arrived"
                if room == "room1":
                    status_message = "Arrived at Room 1"
                elif room == "room2":
                    status_message = "Arrived at Room 2"
                elif room == "nurse_station":
                    status_message = "Arrived at Nurse Station"

        elif state == "arrived":
            if room in ["room1", "room2"]:
                print("\nNavigation worker: starting face detection")

                with lock:
                    current_state = "face_detection"
                    status_message = "Searching for patient..."

                start_face_and_servo()
                time.sleep(3)

                with lock:
                    current_state = "ready_for_zoom"
                    status_message = "Patient verified. Ready for Zoom."
                    patient_verified = True
                    zoom_ready = True

            elif room == "nurse_station":
                stop_face_and_servo()
                with lock:
                    current_state = "waiting"
                    status_message = "Waiting for nurse command"
                    patient_verified = False
                    zoom_ready = False

        elif state == "zoom_active":
            pass

        time.sleep(0.2)


def get_char():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def start_background_threads():
    global lidar_thread_started, lidar_output_thread_started, nav_thread_started

    if not lidar_thread_started:
        lidarThread = threading.Thread(target=runLidar, daemon=True)
        lidarThread.start()
        lidar_thread_started = True
        print("Lidar thread started")

    if not lidar_output_thread_started:
        lidarOutputThread = threading.Thread(target=runLidarOutput, daemon=True)
        lidarOutputThread.start()
        lidar_output_thread_started = True
        print("Lidar output thread started")

    if not nav_thread_started:
        navThread = threading.Thread(target=navigation_worker, daemon=True)
        navThread.start()
        nav_thread_started = True
        print("Navigation worker started")


if __name__ == "__main__":
    zoomUI.start()
    start_background_threads()

    print("\nControls:")
    print("  1 -> send to room1")
    print("  2 -> send to room2")
    print("  N -> send to nurse station")
    print("  A -> start motors")
    print("  W -> forward")
    print("  S -> backward")
    print("  Q -> right spin")
    print("  E -> left spin")
    print("  D -> stop move")
    print("  L -> start lidar thread")
    print("  K -> start lidar output thread")
    print("  F -> start face + servo")
    print("  G -> print face result")
    print("  H -> stop face + servo")
    print("  Z -> mark zoom active")
    print("  R -> reset coords")
    print("  J -> quit safely\n")

    while True:
        c = get_char().upper()

        if c == "1":
            handle_room_command("room1")

        elif c == "2":
            handle_room_command("room2")

        elif c == "N":
            handle_room_command("nurse_station")

        elif c == "A":
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
            if not lidar_thread_started:
                lidarThread = threading.Thread(target=runLidar, daemon=True)
                lidarThread.start()
                lidar_thread_started = True
            else:
                print("Lidar thread already running")

        elif c == "K":
            print("START LIDAR OUTPUT THREAD")
            if not lidar_output_thread_started:
                lidarOutputThread = threading.Thread(target=runLidarOutput, daemon=True)
                lidarOutputThread.start()
                lidar_output_thread_started = True
            else:
                print("Lidar output thread already running")

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
            start_face_and_servo()

        elif c == "G":
            if faceThread is not None:
                result = faceThread.get_result()
                print("FACE SEEN:", result["face_seen"], "BBOX:", result["bbox"])
            else:
                print("Face thread not started")

        elif c == "H":
            print("STOP FACE DETECTION + SERVO TRACKING")
            stop_face_and_servo()

        elif c == "Z":
            print("ZOOM ACTIVE")
            handle_room_command("start_zoom")

        elif c == "R":
            print("RESETTING COORDS")
            with open(CORD_SAVE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "x": str(float(5000.0)),
                        "y": str(float(5000.0)),
                        "theta": str(float(0.0))
                    },
                    f
                )
            break

        elif c == "J":
            print("STOPPED MOTORS")
            m.RUN_MOTORS = False
            m.motorStop(0)

            stop_face_and_servo()

            with open(CORD_SAVE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "x": str(float(SAVED_X)),
                        "y": str(float(SAVED_Y)),
                        "theta": str(float(SAVED_THETA))
                    },
                    f
                )

            try:
                pi.set_PWM_dutycycle(m.MOTOR1_GPIO, 0)
                pi.set_PWM_dutycycle(m.MOTOR2_GPIO, 0)
            except Exception:
                pass

            try:
                lidar.stop()
                lidar.disconnect()
            except Exception:
                pass

            pi.stop()
            break

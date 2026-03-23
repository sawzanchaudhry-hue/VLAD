#!/usr/bin/env python3

import threading
import time
import pigpio

SERVO_GPIO = 4

# Fixed positions from high tilt to low tilt
# Smaller angle = iPad tilts up
# Larger angle = iPad tilts down
SERVO_POSITIONS = [10, 45, 80, 115, 150]

# Start in the middle position
DEFAULT_INDEX = 2

SERVO_MIN_ANGLE = min(SERVO_POSITIONS)
SERVO_MAX_ANGLE = max(SERVO_POSITIONS)

FACE_TIMEOUT = 0.8
LOOP_DELAY = 0.12

# How many consecutive loops a face must stay in a zone before moving
REQUIRED_COUNT_TO_MOVE = 3

# How many consecutive loops centered before locking
REQUIRED_COUNT_TO_LOCK = 6

# Zone boundaries for vertical face center in the image
# For a 480 px tall frame, these are reasonable starting values
TOP_ZONE = 160
CENTER_TOP = 210
CENTER_BOTTOM = 270
BOTTOM_ZONE = 320


class ServoTrackingThread(threading.Thread):
    def __init__(self, face_thread):
        super().__init__(daemon=True)
        self.face_thread = face_thread
        self.running = True

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon")

        self.pi.set_mode(SERVO_GPIO, pigpio.OUTPUT)

        self.current_index = DEFAULT_INDEX
        self.current_angle = SERVO_POSITIONS[self.current_index]

        self.locked = False
        self.up_count = 0
        self.down_count = 0
        self.center_count = 0

        self.set_servo_angle(self.current_angle)

    def set_servo_angle(self, angle):
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, float(angle)))
        pulse_width = int(500 + (angle / 180.0) * 2000)
        self.pi.set_servo_pulsewidth(SERVO_GPIO, pulse_width)
        self.current_angle = angle

    def move_to_index(self, new_index):
        new_index = max(0, min(len(SERVO_POSITIONS) - 1, new_index))
        if new_index != self.current_index:
            self.current_index = new_index
            new_angle = SERVO_POSITIONS[self.current_index]
            print(f"Servo step -> index {self.current_index}, angle {new_angle}")
            self.set_servo_angle(new_angle)

    def run(self):
        print("ServoTrackingThread started")

        while self.running:
            if self.face_thread is None:
                time.sleep(LOOP_DELAY)
                continue

            result = self.face_thread.get_result()
            face_seen = result["face_seen"]
            bbox = result["bbox"]
            frame = result["frame"]
            last_time = result["timestamp"]

            recently_seen = (time.time() - last_time) < FACE_TIMEOUT
            valid_face = face_seen and recently_seen and bbox is not None and frame is not None

            if not valid_face:
                self.up_count = 0
                self.down_count = 0
                self.center_count = 0
                self.locked = False
                time.sleep(LOOP_DELAY)
                continue

            x1, y1, x2, y2 = bbox
            frame_h, frame_w = frame.shape[:2]
            face_center_y = (y1 + y2) / 2.0

            # If already locked, do nothing
            if self.locked:
                time.sleep(LOOP_DELAY)
                continue

            # Decide zone
            if face_center_y < TOP_ZONE:
                # Face is too high in frame -> tilt iPad up -> smaller angle -> lower index
                self.up_count += 1
                self.down_count = 0
                self.center_count = 0

                if self.up_count >= REQUIRED_COUNT_TO_MOVE:
                    self.move_to_index(self.current_index - 1)
                    self.up_count = 0

            elif face_center_y > BOTTOM_ZONE:
                # Face is too low in frame -> tilt iPad down -> larger angle -> higher index
                self.down_count += 1
                self.up_count = 0
                self.center_count = 0

                if self.down_count >= REQUIRED_COUNT_TO_MOVE:
                    self.move_to_index(self.current_index + 1)
                    self.down_count = 0

            elif CENTER_TOP <= face_center_y <= CENTER_BOTTOM:
                # Face is centered enough
                self.center_count += 1
                self.up_count = 0
                self.down_count = 0

                if self.center_count >= REQUIRED_COUNT_TO_LOCK:
                    self.locked = True
                    print(f"Servo locked at index {self.current_index}, angle {self.current_angle}")

            else:
                # In between zones: don't move, just reset move counters a bit
                self.up_count = 0
                self.down_count = 0
                self.center_count = 0

            time.sleep(LOOP_DELAY)

    def unlock(self):
        print("Servo unlocked")
        self.locked = False
        self.up_count = 0
        self.down_count = 0
        self.center_count = 0

    def stop(self):
        print("ServoTrackingThread stopping")
        self.running = False
        try:
            self.set_servo_angle(SERVO_POSITIONS[DEFAULT_INDEX])
            time.sleep(0.2)
            self.pi.set_servo_pulsewidth(SERVO_GPIO, 0)
            self.pi.stop()
        except Exception:
            pass

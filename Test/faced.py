#!/usr/bin/env python3

import threading
import time
import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork


class FaceDetectionThread(threading.Thread):
    def __init__(self, show_window=True):
        super().__init__(daemon=False)
        self.running = True
        self.show_window = show_window
        self.lock = threading.Lock()

        self.face_seen = False
        self.latest_bbox = None
        self.latest_frame = None
        self.last_update = 0.0

    def stop(self):
        print("FaceDetectionThread.stop() called")
        self.running = False

    def get_result(self):
        with self.lock:
            return {
                "face_seen": self.face_seen,
                "bbox": self.latest_bbox,
                "timestamp": self.last_update,
                "frame": None if self.latest_frame is None else self.latest_frame.copy()
            }

    def run(self):
        print("FaceDetectionThread.run() started")

        try:
            with dai.Pipeline() as pipeline:
                # v3 camera node
                cam = pipeline.create(dai.node.Camera).build()

                # Separate display/debug stream
                # This is just for your host-side frame display/storage
                rgb = cam.requestOutput(
                    size=(640, 480),
                    type=dai.ImgFrame.Type.BGR888p,
                    fps=15
                )

                # v3 model + parser setup
                # YuNet 320x240 is an official model variant
                model = "luxonis/yunet:320x240"

                face_nn = pipeline.create(ParsingNeuralNetwork).build(
                    cam,
                    model,
                    fps=10
                )

                q_rgb = rgb.createOutputQueue(maxSize=1, blocking=False)
                q_det = face_nn.out.createOutputQueue(maxSize=1, blocking=False)

                print("Starting OAK-D pipeline...")
                pipeline.start()
                print("OAK-D pipeline started")

                while self.running and pipeline.isRunning():
                    in_rgb = q_rgb.tryGet()
                    in_det = q_det.tryGet()

                    if in_rgb is None:
                        time.sleep(0.01)
                        continue

                    frame = in_rgb.getCvFrame()
                    h, w = frame.shape[:2]

                    with self.lock:
                        face_seen = self.face_seen
                        best_bbox = self.latest_bbox
                        last_time = self.last_update

                    # Only update detection state when a new parsed packet arrives
                    if in_det is not None:
                        detections = in_det.detections
                        #print("detections:", len(detections))

                        if len(detections) > 0:
                            # Pick highest-confidence face
                            det = max(detections, key=lambda d: d.confidence)

                            x1 = int(det.xmin * w)
                            y1 = int(det.ymin * h)
                            x2 = int(det.xmax * w)
                            y2 = int(det.ymax * h)

                            # Clamp just in case
                            x1 = max(0, min(x1, w - 1))
                            y1 = max(0, min(y1, h - 1))
                            x2 = max(0, min(x2, w - 1))
                            y2 = max(0, min(y2, h - 1))

                            face_seen = True
                            best_bbox = (x1, y1, x2, y2)
                            last_time = time.time()

                            #print("FACE DETECTED:", best_bbox, "conf =", det.confidence)
                        else:
                            # Clear only after short timeout
                            if time.time() - last_time > 0.5:
                                face_seen = False
                                best_bbox = None

                    if best_bbox is not None:
                        x1, y1, x2, y2 = best_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            "Face",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                    with self.lock:
                        self.face_seen = face_seen
                        self.latest_bbox = best_bbox
                        self.latest_frame = frame.copy() if best_bbox is not None else None
                        self.last_update = last_time

                    # Only use this if your OpenCV supports GUI windows
                    if self.show_window:
                        cv2.imshow("OAK-D Face Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.running = False

                    time.sleep(0.01)

        except Exception as e:
            print("Face detection thread error:", repr(e))

        finally:
            print("FaceDetectionThread exiting")
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                except Exception:
                    pass

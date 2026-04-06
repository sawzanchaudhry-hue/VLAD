#!/usr/bin/env python3

import threading
import time
import depthai as dai
import numpy as np


class BedDetectionThread(threading.Thread):
    def __init__(self, show_all_labels=False, min_confidence=0.40):
        super().__init__(daemon=False)
        self.running = True
        self.lock = threading.Lock()

        self.show_all_labels = show_all_labels
        self.min_confidence = min_confidence

        self.bed_seen = False
        self.latest_bed_bbox = None
        self.latest_frame = None
        self.last_update = 0.0
        self.latest_label = None
        self.latest_confidence = 0.0

    def stop(self):
        print("BedDetectionThread.stop() called")
        self.running = False

    def get_result(self):
        with self.lock:
            return {
                "bed_seen": self.bed_seen,
                "bed_bbox": self.latest_bed_bbox,
                "timestamp": self.last_update,
                "frame": None if self.latest_frame is None else self.latest_frame.copy(),
                "label": self.latest_label,
                "confidence": self.latest_confidence,
            }

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def run(self):
        print("BedDetectionThread.run() started")

        try:
            with dai.Pipeline() as pipeline:
                cameraNode = pipeline.create(dai.node.Camera).build()

                detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
                    cameraNode,
                    dai.NNModelDescription("yolov6-nano")
                )

                labelMap = detectionNetwork.getClasses()

                qRgb = detectionNetwork.passthrough.createOutputQueue()
                qDet = detectionNetwork.out.createOutputQueue()

                print("Starting OAK-D bed pipeline...")
                pipeline.start()
                print("OAK-D bed pipeline started")

                frame = None
                detections = []

                while self.running and pipeline.isRunning():
                    inRgb = qRgb.tryGet()
                    inDet = qDet.tryGet()

                    if inRgb is not None:
                        frame = inRgb.getCvFrame()

                    if inDet is not None:
                        detections = inDet.detections

                    best_bed = None
                    best_conf = 0.0
                    best_label = None

                    if frame is not None and detections:
                        for detection in detections:
                            label = labelMap[detection.label] if detection.label < len(labelMap) else "Unknown"
                            confidence = float(detection.confidence)

                            if self.show_all_labels:
                                print(f"Detected: {label} ({confidence:.2f})")

                            if label.lower() == "bed" and confidence >= self.min_confidence:
                                bbox = self.frameNorm(
                                    frame,
                                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                )

                                if confidence > best_conf:
                                    best_conf = confidence
                                    best_bed = tuple(bbox.tolist())
                                    best_label = label

                    with self.lock:
                        if best_bed is not None:
                            self.bed_seen = True
                            self.latest_bed_bbox = best_bed
                            self.latest_frame = frame.copy() if frame is not None else None
                            self.last_update = time.time()
                            self.latest_label = best_label
                            self.latest_confidence = best_conf
                        else:
                            if time.time() - self.last_update > 0.5:
                                self.bed_seen = False
                                self.latest_bed_bbox = None
                                self.latest_frame = frame.copy() if frame is not None else None
                                self.latest_label = None
                                self.latest_confidence = 0.0

                    time.sleep(0.01)

        except Exception as e:
            print("Bed detection thread error:", repr(e))

        finally:
            print("BedDetectionThread exiting")

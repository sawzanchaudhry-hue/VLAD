import pigpio
import open3d as o3d
import small_gicp
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import sys
import tty
import threading
from threading import Lock
import queue
import math
from rplidar import RPLidar, RPLidarException
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import RPLidarA1
from breezyslam.sensors import Laser
import numpy as np
import multiprocessing as mp
from multiprocessing import Process
#import matplotlib
#matplotlib.use('Agg')  # no display needed
#import matplotlib.pyplot as plt
#mp.set_start_method('spawn')
from mrpt.pymrpt import mrpt
#from PIL import Image
import heapq

MAP_PIXEL_SIZE = 4000
MAP_ROOM_SIZE = 5

VOXEL_SIZE = 0.05
MIN_POINTS = 20

class Lidar:
    def __init__(self, m):
        #Test
        #Global
        self.globalPosX = 0
        self.globalPosY = 0
        self.globalPosTheta = 0
        self.scanComplete = threading.Event()
        self.prevWasForward = True
        self.attemptsTurning = 0
        self.attemptsForward = 0
        self.prevWheelEncoderCount = 0
        #Motor Class
        self.m = m
        self.motorEncoderCountFunc = self.m.motorEncoderCountForLIDAR
        #Global Variables
        self.lastUpdateTime = time.time()
        self.lastEncoderCount = self.m.motorEncoderCountForLIDAR()
        self.lastAngleCount = self.m.motorAngleForLIDAR()
        #Lidar
        self.PORT = "/dev/ttyUSB0"
        self.lidarObj = RPLidar(self.PORT, baudrate=115200, timeout=1)
        self.lidarSettings = Laser(
            scan_size=360,
            scan_rate_hz=10,
            detection_angle_degrees=360,
            distance_no_detection_mm=12000,
            detection_margin=0,
            offset_mm=0
        )
        #BreezySLAM
        self.slamBreezy = RMHC_SLAM(self.lidarSettings, MAP_PIXEL_SIZE, MAP_ROOM_SIZE)
        self.mapBreezy = bytearray(MAP_PIXEL_SIZE * MAP_PIXEL_SIZE)
        #MRPT
        self.localMRPT = None
        self.getStaticMap = False
        self.pf = None
        self.builderMRPT = mrpt.slam.CMetricMapBuilderICP()
        self.builderMRPT.ICP_options.insertionLinDistance    = 0.30
        self.builderMRPT.ICP_options.insertionAngDistance    = 0.15
        self.builderMRPT.ICP_options.localizationLinDistance = 0.01
        self.builderMRPT.ICP_options.localizationAngDistance = 0.01
        self.builderMRPT.ICP_options.matchAgainstTheGrid     = True
        self.builderMRPT.ICP_options.minICPgoodnessToAccept  = 0.90
        self.cfg = mrpt.config.CConfigFileMemory()
        self.cfg.write("MappingApplication", "occupancyGrid_count", 1)
        self.cfg.write("MappingApplication", "occupancyGrid_00_resolution", 0.02)
        self.cfg.write("MappingApplication", "occupancyGrid_00_min_x", -5.0)
        self.cfg.write("MappingApplication", "occupancyGrid_00_max_x",  5.0)
        self.cfg.write("MappingApplication", "occupancyGrid_00_min_y", -5.0)
        self.cfg.write("MappingApplication", "occupancyGrid_00_max_y",  5.0)
        self.cfg.write("MappingApplication", "pointsMap_count", 1)
        self.cfg.write("MappingApplication", "pointsMap_00_insertionOpts_minDistBetweenLaserPoints", 0.05)
        self.builderMRPT.ICP_options.mapInitializers.loadFromConfigFile(self.cfg, "MappingApplication")
        self.builderMRPT.ICP_options.insertionLinDistance    = 0.30
        self.builderMRPT.ICP_options.insertionAngDistance    = 0.15
        self.builderMRPT.ICP_options.localizationLinDistance = 0.01
        self.builderMRPT.ICP_options.localizationAngDistance = 0.01
        self.builderMRPT.ICP_options.matchAgainstTheGrid     = True
        self.builderMRPT.ICP_options.minICPgoodnessToAccept  = 0.90
        #Lidar Threading
        self.runLidarThread = threading.Thread(target=self.runLidar, daemon=True)
        #Open3D
        self.globalPose = np.eye(4)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stopLidar()

    def startLidar(self):
        print("---STARTING LIDAR---")
        self.runLidarThread.start()
    
    def stopLidar(self):
        self.lidarObj.stop()
        if self.getStaticMap == True:
            p = self.builderMRPT.getCurrentPoseEstimation().getMeanVal()
            np.save("SavedInitialMap/LastKnownPose.npy", np.array([p.x(), p.y(), p.yaw()]))
        else:
            p = self.pf.getMostLikelyParticle()
            x = p.x
            y = p.y
            angle = p.phi
            np.save("SavedInitialMap/LastKnownPose.npy", np.array([x, y, angle]))
        print("---SAVED LAST KNOWN POSE---")
        #self.lidarObj.disconnect()

    #This just gets data from the Lidar, organises/sorts it, and then passes it to runMapping() funciton
    def runLidar(self):
        self.lidarObj.start_motor()
        self.lidarObj.clean_input()
        time.sleep(2)
        while True:
            try:
                self.m.RUN_MOTORS = True
                scanNum = 0
                for scan in self.lidarObj.iter_scans(max_buf_meas=4250, scan_type="normal"):
                    scanNum += 1
                    if scanNum % 3 != 0:
                        continue
                    scanNum = 0
                    angleDistances = [0] * 360
                    angleDistances2 = [0] * 360
                    scanPointsXY = []
                    scanOut = scan
                    for data in scanOut:
                        currentAngle = int(data[1]) #% 360
                        if (0 <= currentAngle < 90) or (270 <= currentAngle < 360):
                            if angleDistances[currentAngle] == 0 or data[2] < angleDistances[currentAngle]:
                                angleDistances[currentAngle] = data[2] / 1000.0 #For Breezy remove / 1000.0
                                angleDistances2[currentAngle] = data[2]
                                #angleRad = math.radians(currentAngle)
                                #x = data[2] * math.cos(angleRad)
                                #y = data[2] * math.sin(angleRad)
                                #scanPointsXY.append((x, y))
                    scanPointsArray = np.array(scanPointsXY)
                    #For MRPT
                    if self.getStaticMap == True:
                        #self.runStaticMapping(angleDistances)
                        i = 0
                    else:
                        #self.runRegularMapping(angleDistances)
                        i = 0
                    #For BreezySLAM
                    self.runMapping(angleDistances2, scanPointsArray)
            except RPLidarException as e:
                print("---LIDAR CRASHED... GOING TO RETRY---")
                print(f"Reason: {e}")
                self.m.RUN_MOTORS = False
                self.lidarObj.stop()
                time.sleep(0.5)
                self.lidarObj.start_motor()
                self.lidarObj.clean_input()
                time.sleep(1)
                continue

    def runRegularMapping(self, lidarOutput):
        #Handle LIDAR Data
        obsMRPT = mrpt.obs.CObservation2DRangeScan()
        obsMRPT.aperture    = 2 * math.pi
        obsMRPT.maxRange    = 12.0
        obsMRPT.rightToLeft = True
        sensorPose = mrpt.poses.CPose3D(0.336, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, yaw, pitch, roll
        obsMRPT.setSensorPose(sensorPose)
        obsMRPT.resizeScan(360)
        for j, r in enumerate(lidarOutput):
            if 120 <= j <= 240:
                obsMRPT.setScanRange(j, 0.0)
                obsMRPT.setScanRangeValidity(j, False)
            else:
                obsMRPT.setScanRange(j, float(r))
                obsMRPT.setScanRangeValidity(j, r > 0.0)
        #Odomotery from Wheel Encoders (Zero for now)
        actions = mrpt.obs.CActionCollection()
        movement = mrpt.obs.CActionRobotMovement2D()
        motionModel = mrpt.obs.CActionRobotMovement2D.TMotionModelOptions()
        motionModel.modelSelection = mrpt.obs.CActionRobotMovement2D.TDrawSampleMotionModel.mmGaussian
        motionModel.gaussianModel.minStdXY  = 0.005
        motionModel.gaussianModel.minStdPHI = 0.1
        distanceMoved = ((self.m.motorEncoderCountForLIDAR() - self.lastEncoderCount) / 340)
        angleDiff = math.radians(self.m.motorAngleForLIDAR() - self.lastAngleCount)
        p = self.pf.getMostLikelyParticle()
        x     = p.x
        y     = p.y
        current_angle = math.degrees(p.phi)
        dx = distanceMoved * math.cos(current_angle)
        dy = distanceMoved * math.sin(current_angle)
        if self.m.ANGLE_TURNING == True:
            dx = 0
            dy = 0
            angleDiff = 0
        else:
            angleDiff = 0
        if self.m.TURNING_RIGHT == False:
            angleDiff = angleDiff * -1
        movement.computeFromOdometry(mrpt.poses.CPose2D(dx, dy, angleDiff), motionModel)
        actions.insert(movement)
        self.lastEncoderCount = self.m.motorEncoderCountForLIDAR()
        self.lastAngleCount = self.m.motorAngleForLIDAR()
        #
        obs = mrpt.obs.CSensoryFrame()
        obs.insert(obsMRPT)
        pfOptions = mrpt.bayes.CParticleFilter.TParticleFilterOptions()
        self.pf.prediction_and_update(actions, obs, pfOptions)
        p = self.pf.getMostLikelyParticle()
        x     = p.x
        y     = p.y
        angle = math.degrees(p.phi)
        print(f"\rx={x}, y={y}, angle={angle}", end="", flush=True)
        self.runLogic(lidarOutput)

    def runStaticMapping(self, lidarOutput):
        #Handle Motor Encoder Data
        """
        odom = mrpt.obs.CObservationOdometry()
        distanceMoved = ((self.m.motorEncoderCountForLIDAR() - self.lastEncoderCount) / 340) * 1000.0
        angleDiff = math.radians(self.m.motorAngleForLIDAR() - self.lastAngleCount)
        print(f"YOYO COUNT: {distanceMoved}")
        print(f"YOYO TURN: {angleDiff}")
        p = self.builderMRPT.getCurrentPoseEstimation().getMeanVal()
        current_angle = p.yaw()
        dx = distanceMoved * math.cos(current_angle)
        dy = distanceMoved * math.sin(current_angle)
        if self.m.ANGLE_TURNING == True:
            dx = 0
            dy = 0
        else:
            angleDiff = 0
        odom.odometry = mrpt.poses.CPose2D(dx, dy, angleDiff)
        self.builderMRPT.processObservation(odom)
        self.lastEncoderCount = self.m.motorEncoderCountForLIDAR()
        self.lastAngleCount = self.m.motorAngleForLIDAR()
        """
        #Handle LIDAR Data
        obsMRPT = mrpt.obs.CObservation2DRangeScan()
        obsMRPT.aperture    = 2 * math.pi
        obsMRPT.maxRange    = 12.0
        obsMRPT.rightToLeft = True
        sensorPose = mrpt.poses.CPose3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, yaw, pitch, roll
        obsMRPT.setSensorPose(sensorPose)
        obsMRPT.resizeScan(360)
        for j, r in enumerate(lidarOutput):
            if False: #120 <= j <= 240:
                obsMRPT.setScanRange(j, 0.0)
                obsMRPT.setScanRangeValidity(j, False)
            else:
                obsMRPT.setScanRange(j, float(r))
                obsMRPT.setScanRangeValidity(j, r > 0.0)
        self.builderMRPT.processObservation(obsMRPT)
        p = self.builderMRPT.getCurrentPoseEstimation().getMeanVal()
        x     = p.x()
        y     = p.y()
        angle = math.degrees(p.yaw())
        #print(f"\rx={x}, y={y}, angle={angle}", end="", flush=True)

    def runMapping(self, lidarOutput, scanPointsArray):
        #This part just uses BreezySLAM
        #angleDistancesFixed = lidarOutput[0:90] + lidarOutput[270:360]
        #timeDiff = time.time() - self.lastUpdateTime
        #encoderCount = ((self.m.motorEncoderCountForLIDAR() - self.lastEncoderCount) / 340) * 1000.0
        #print(f"encoderCOUNT={self.motorEncoderCountFunc()}")
        #self.slamBreezy.update(lidarOutput, pose_change=(encoderCount, 0, timeDiff))
        self.slamBreezy.update(lidarOutput)
        #self.lastUpdateTime = time.time()
        #self.lastEncoderCount = self.m.motorEncoderCountForLIDAR()
        bx, by, btheta = self.slamBreezy.getpos()
        btheta = btheta % 360
        self.globalPosX = bx; self.globalPosY = by; self.globalPosTheta = btheta
        self.scanComplete.set()
        print(f"\rx={bx}, y={by}, angle={btheta}", end="", flush=True)
        self.runLogic(lidarOutput, scanPointsArray)

    def runLogic(self, lidarOutput, scanPointsArray):
        atleastOneObject = False
        for angle, dist in enumerate(lidarOutput):
            if (0 <= angle < 90) or (270 <= angle < 360):
                #print(f"Angle: {angle}")
                if dist <= 400 and dist != 0:
                    atleastOneObject = True
                else:
                    oo = 0
        if atleastOneObject == True:
            print("---OBJECT DETECTED----")
            self.m.pauseMotorEvent.clear()
            self.m.pauseMotor(0)
        else:
            self.m.pauseMotorEvent.set()
        """
        if self.m.MOTOR_FINISHED == False:
            if atleastOneObject == True and self.m.ANGLE_TURNING == False:
                self.m.forceMotorStop(0)
                self.m.RUN_MOTORS = True
            if (self.m.MOTOR_ENCODER_LIDAR2 - self.prevWheelEncoderCount) == 0:
                ii = self.m.MOTOR_ENCODER_LIDAR2 - self.prevWheelEncoderCount
                print(f"encod: {self.m.MOTOR_ENCODER_LIDAR2}")
                if self.attemptsForward >= 5:
                    self.m.forceMotorStop(0)
                    self.m.RUN_MOTORS = True
                    self.m.motorRunQueue.put(("B", 0.3))
                self.attemptsForward += 1
        if self.m.MOTOR_FINISHED == True:
            if atleastOneObject == True: #self.globalPosTheta
                self.m.motorRunQueue.put(("R", 10))
                self.attemptsForward = 0
            else:
                self.m.motorRunQueue.put(("F", 0.25))
                wantedAngleDir = 180 - self.globalPosTheta
                if wantedAngleDir >= 0:
                    self.m.motorRunQueue.put(("R", abs(wantedAngleDir)))
                else:
                    self.m.motorRunQueue.put(("L", abs(wantedAngleDir)))
                self.m.motorRunQueue.put(("F", 1))
                self.attemptsForward = 0
        self.prevWheelEncoderCount = self.m.MOTOR_ENCODER_LIDAR2
        """

        """
        if atleastOneObject == True:
            backOut = False
            if self.attemptsTurning >= 10:
                if (self.m.MOTOR_ENCODER_LIDAR - self.prevWheelEncoderCount) == 0:
                    backOut = True
            if backOut == True:
                self.m.pi.set_servo_pulsewidth(self.m.MOTOR1_GPIO, 1390)
                self.m.pi.set_servo_pulsewidth(self.m.MOTOR2_GPIO, 1610)
                self.attemptsTurning = 0
            else:
                self.m.pi.set_servo_pulsewidth(self.m.MOTOR1_GPIO, 1610)
                self.m.pi.set_servo_pulsewidth(self.m.MOTOR2_GPIO, 1610)
            self.prevWheelEncoderCount = self.m.MOTOR_ENCODER_LIDAR
            self.attemptsForward = 0
            self.attemptsTurning += 1
        else:
            self.m.pi.set_servo_pulsewidth(self.m.MOTOR1_GPIO, 1610)
            self.m.pi.set_servo_pulsewidth(self.m.MOTOR2_GPIO, 1390)
            self.prevWheelEncoderCount = self.m.MOTOR_ENCODER_LIDAR
            self.attemptsTurning = 0
            self.attemptsForward += 1
        """

    def is_valid(state):
        x, y = state[0], state[1]
        for ox, oy in obstacles:
            if (x - ox)**2 + (y - oy)**2 < ROBOT_RADIUS**2:
                return False
        return True

    #Mapping Options (For MRPT)
    def runStaticMap(self):
        self.getStaticMap = True
        #self.builderMRPT.initialize()
        self.startLidar()

    def runRegularMap(self):
        self.getStaticMap = False
        self.loadCompleteMap2()
        self.startLidar()

    #Map Handling
    def saveInitialMap(self):
        print("---SAVING INITIAL MAP---")
        self.slamBreezy.getmap(self.mapBreezy)
        np.save("SavedInitialMap/SavedInitialMapPoints.npy", np.array(self.mapBreezy))
        print("---INITIAL MAP SAVED AS NPY FILE---")

    def saveInitialMap2(self):
        print("---SAVING STATIC INITIAL MAP---")
        simpleMap = mrpt.maps.CSimpleMap()
        self.builderMRPT.getCurrentlyBuiltMap(simpleMap)
        simpleMap.saveToFile("SavedInitialMap/SavedStaticMap.simplemap")
        print("---STATIC INITIAL MAP SAVED AS NPY FILE---")

    def showInitialMap2(self):
        simpleMap = mrpt.maps.CSimpleMap()
        self.builderMRPT.getCurrentlyBuiltMap(simpleMap)
        grid = mrpt.maps.COccupancyGridMap2D()
        grid.loadFromSimpleMap(simpleMap)
        grid.saveAsBitmapFile("map.png")

    def loadCompleteMap(self):
        print("---ATTEMPTING TO LOAD COMPLETE MAP---")
        loadedMap = np.load("SavedInitialMap/SavedInitialMapPoints.npy")
        flattenLoadedMap = loadedMap.astype(np.uint8).flatten()
        self.mapBreezy[:] = flattenLoadedMap.tobytes()
        self.slamBreezy.setmap(self.mapBreezy)
        print("---COMPLETE MAP LOADED---")

    def loadCompleteMap2(self):
        print("---ATTEMPTING TO LOAD COMPLETE STATIC MAP---")
        if os.path.exists("SavedInitialMap/SavedStaticMap.simplemap"):
            #self.localMRPT = MCLLocalizer()
            #self.localMRPT.loadMap("SavedInitialMap/SavedStaticMap.simplemap")
            simpleMap = mrpt.maps.CSimpleMap()
            simpleMap.loadFromFile("SavedInitialMap/SavedStaticMap.simplemap")
            gridMRPT = mrpt.maps.COccupancyGridMap2D()
            gridMRPT.loadFromSimpleMap(simpleMap)
            params = mrpt.slam.TMonteCarloLocalizationParams()
            params.metricMap = gridMRPT
            self.pf = mrpt.slam.CMonteCarloLocalization2D()
            self.pf.options.metricMap = gridMRPT
            self.pf.resetUniformFreeSpace(
                gridMRPT,
                0.3,     # minimum occupancy to be considered free
                1000      # number of particles
            )
        if os.path.exists("SavedInitialMap/LastKnownPose.npy"):
            pose = np.load("SavedInitialMap/LastKnownPose.npy")
            self.pf.resetDeterministic(mrpt.math.TPose2D(float(pose[0]), float(pose[1]), float(pose[2])), 1000)
        print("---STATIC MAP LOADED---")

    def convertInitialMapToImage(self):
        print("---Creating Initial Map Image---")
        mapBytes = np.load("SavedInitialMap/SavedInitialMapPoints.npy")
        grid = mapBytes.reshape((MAP_PIXEL_SIZE, MAP_PIXEL_SIZE))
        #grid = (grid - grid.min()) / (grid.max() - grid.min())
        grid_binary = (grid > 0.55).astype(np.uint8) * 255
        gridSmall = grid[::2, ::2]
        #plt.imshow(grid, cmap="gray", origin="lower") #interpolation="nearest"
        plt.imshow(grid, cmap="gray") #interpolation="nearest"
        #plt.plot(self.globalPosX, self.globalPosY, 'ro', markersize=5)
        plt.savefig("SavedInitialMap/SavedInitialMap.png")
        plt.close()
        print("---Finished Initital Map Image---")

    def convertInitialMapToImage2(self):
        print("---Creating Initial Map Image---")
        mapBytes = np.load("SavedInitialMap/SavedInitialMapPoints.npy")
        grid = mapBytes.reshape((MAP_PIXEL_SIZE, MAP_PIXEL_SIZE))
        im = Image.fromarray(grid)
        im.save("SavedInitialMap/SavedInitialMap.png")
        print("---Finished Initital Map Image---")

    def showMap(self):
        mapbytes = np.load("SavedInitialMap/SavedInitialMapPoints.npy")
        grid = mapbytes.reshape((1000, 1000))
        plt.imshow(grid, cmap='gray')
        plt.savefig("SavedInitialMap/SavedInitialMap.png")
        plt.close()
        print("Saved map.png")

    def showMap2(self):
        self.slamBreezy.getmap(self.mapBreezy)
        grid = np.array(self.mapBreezy).reshape((MAP_PIXEL_SIZE, MAP_PIXEL_SIZE))
        plt.clf()
        plt.imshow(grid, cmap='gray')
        plt.pause(1)

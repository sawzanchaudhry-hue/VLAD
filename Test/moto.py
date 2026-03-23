import pigpio
import time
import sys
import tty
import threading
import queue

class Motor:
    def __init__(self, pi):
        self.pi = pi
        self.motorRunQueue = queue.Queue()
        self.pauseMotorEvent = threading.Event()
        self.pauseMotorEvent.set()
        #For LIDAR
        self.ANGLE_TURNING = False
        self.TURNING_RIGHT = False
        self.ANGLE_COUNT = 0
        self.MOTOR_FINISHED = True
        #Constant Variables
        self.RUN_MOTORS = False
        self.piValue = 3.1416
        self.TOTAL_ENCODER_COUNT = 0
        self.MOTOR_ENCODER_LIDAR = 0
        self.MOTOR_ENCODER_LIDAR2 = 0
        #Motor 1 Global Variables
        self.MOTOR1_GPIO = 12
        self.MOTOR1_A_GPIO = 23
        self.MOTOR1_B_GPIO = 24
        self.MOTOR1_PERCENT = 0
        self.MOTOR1_A_COUNT = 0
        self.MOTOR1_ENCODER_COUNT = 0
        #Motor 2 Global Variables
        self.MOTOR2_GPIO = 13
        self.MOTOR2_A_GPIO = 5
        self.MOTOR2_B_GPIO = 6
        self.MOTOR2_PERCENT = 0
        self.MOTOR2_A_COUNT = 0
        self.MOTOR2_ENCODER_COUNT = 0
        #Extra
        self.MOTOR1_GRAPH = []
        self.MOTOR2_GRAPH = []
        #Set Motor 1 GPIO
        self.pi.set_mode(self.MOTOR1_GPIO, pigpio.OUTPUT)
        self.pi.set_mode(self.MOTOR1_A_GPIO, pigpio.INPUT)
        self.pi.set_mode(self.MOTOR1_B_GPIO, pigpio.INPUT)
        self.pi.set_pull_up_down(self.MOTOR1_A_GPIO, pigpio.PUD_UP)
        self.pi.set_pull_up_down(self.MOTOR1_B_GPIO, pigpio.PUD_UP)
        #Set Motor 2 GPIO
        self.pi.set_mode(self.MOTOR2_GPIO, pigpio.OUTPUT)
        self.pi.set_mode(self.MOTOR2_A_GPIO, pigpio.INPUT)
        self.pi.set_mode(self.MOTOR2_B_GPIO, pigpio.INPUT)
        self.pi.set_pull_up_down(self.MOTOR2_A_GPIO, pigpio.PUD_UP)
        self.pi.set_pull_up_down(self.MOTOR2_B_GPIO, pigpio.PUD_UP)
        #Run Encoder Collection
        motor1Collector = self.pi.callback(self.MOTOR1_A_GPIO, pigpio.EITHER_EDGE, self.motor1EncoderCollect)
        motor2Collector = self.pi.callback(self.MOTOR2_A_GPIO, pigpio.EITHER_EDGE, self.motor2EncoderCollect)
        #Set Both Motor PWM Frequency
        self.PWM_FREQ = 50
        self.pi.set_PWM_frequency(self.MOTOR1_GPIO, self.PWM_FREQ)
        self.pi.set_PWM_frequency(self.MOTOR2_GPIO, self.PWM_FREQ)
        #Run Motor Threading
        self.motorThread = threading.Thread(target=self.runMotorThread, daemon=True)
        self.motorThread.start()
        #Run Motor Queue
        self.motorQueue = threading.Thread(target=self.runMotorQueue, args=(self.motorRunQueue,), daemon=True)
        self.motorQueue.start()
        
    #For LIDAR
    def motorEncoderCountForLIDAR(self):
        return self.MOTOR1_A_COUNT

    def motorAngleForLIDAR(self):
        return self.ANGLE_COUNT

    #Encoder Handling
    def motor1EncoderCollect(self, gpio, level, tick):
        #self.TOTAL_ENCODER_COUNT += 1
        if level == 1:
            self.MOTOR1_A_COUNT += 1
            self.MOTOR_ENCODER_LIDAR += 1
            self.MOTOR_ENCODER_LIDAR2 += 1
            if self.ANGLE_TURNING == True:
                self.ANGLE_COUNT += 0.45
        
    def motor2EncoderCollect(self, gpio, level, tick):
        #self.TOTAL_ENCODER_COUNT += 1
        if level == 1:
            self.MOTOR2_A_COUNT += 1
            
    #Motor Threading
    def motorThreadFunc(self):
        self.MOTOR1_GRAPH.append(self.MOTOR1_A_COUNT)
        self.MOTOR2_GRAPH.append(self.MOTOR2_A_COUNT)
    
    def runMotorThread(self):
        while True:
            if self.RUN_MOTORS == True:
                #self.MOTOR1_A_COUNT = 0
                #self.MOTOR2_A_COUNT = 0
                start = time.time()
                elapsed = time.time() - start
                time.sleep(max(0, 0.1 - elapsed))
                self.motorThreadFunc()

    def runMotorQueue(self, runMotorQueue):
        while True:
            if self.RUN_MOTORS == True:
                command = runMotorQueue.get()
                direction, amount = command
                if direction == "F":
                    self.motorForward(amount)
                elif direction == "B":
                    self.motorBackward(amount)
                elif direction == "R":
                    self.motorRightSpin(amount)
                elif direction == "L":
                    self.motorLeftSpin(amount)
                elif direction == "S":
                    self.motorStop(amount)

    #Motor Calculations          
    def getRevolutionsFromDistance(self, distance):
        return(340 * distance)

    #For LIDAR
    def encoderCountForLIDAR(self):
        forLIDAR = self.MOTOR_ENCODER_LIDAR
        self.MOTOR_ENCODER_LIDAR = 0
        return(forLIDAR / 400)

    #Motor Movement Functions
    def motorForward(self, AMOUNT):
        self.RUN_MOTORS = True
        self.MOTOR_FINISHED = False
        self.ANGLE_TURNING = False
        self.MOTOR1_A_COUNT = 0
        self.MOTOR2_A_COUNT = 0
        currentEncoderCount = self.MOTOR1_A_COUNT
        finalAmount = self.getRevolutionsFromDistance(AMOUNT)
        while ((self.MOTOR1_A_COUNT - currentEncoderCount) <= finalAmount) and (self.RUN_MOTORS == True):
            self.pauseMotorEvent.wait()
            self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1610) #1610
            self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1390) #1390
        self.MOTOR_FINISHED = True
        self.motorStop(0)
        
    def motorBackward(self, AMOUNT):
        self.RUN_MOTORS = True
        self.MOTOR_FINISHED = False
        self.ANGLE_TURNING = False
        self.MOTOR1_A_COUNT = 0
        self.MOTOR2_A_COUNT = 0
        currentEncoderCount = self.MOTOR1_A_COUNT
        finalAmount = self.getRevolutionsFromDistance(AMOUNT)
        while ((self.MOTOR1_A_COUNT - currentEncoderCount) <= finalAmount) and (self.RUN_MOTORS == True):
            self.pauseMotorEvent.wait()
            self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1390) #1390
            self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1610) #1610
        self.MOTOR_FINISHED = True
        self.motorStop(0)

    def motorRightSpin(self, AMOUNT):
        self.RUN_MOTORS = True
        self.MOTOR_FINISHED = False
        self.MOTOR1_A_COUNT = 0
        self.MOTOR2_A_COUNT = 0
        self.ANGLE_COUNT = 0
        currentEncoderCount = self.MOTOR1_A_COUNT
        currentEncoderCount2 = self.MOTOR2_A_COUNT
        finalAmount = 350 * (AMOUNT / 180)
        self.ANGLE_TURNING = True
        self.TURNING_RIGHT = True
        while (((self.MOTOR1_A_COUNT - currentEncoderCount) <= finalAmount) or ((self.MOTOR2_A_COUNT - currentEncoderCount2) <= finalAmount)) and (self.RUN_MOTORS == True):
            self.pauseMotorEvent.wait()
            self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1610) #1610
            self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1610)
        self.ANGLE_TURNING = False
        self.MOTOR_FINISHED = True
        self.motorStop(0)
        
    def motorLeftSpin(self, AMOUNT):
        self.RUN_MOTORS = True
        self.MOTOR_FINISHED = False
        self.MOTOR1_A_COUNT = 0
        self.MOTOR2_A_COUNT = 0
        self.ANGLE_COUNT = 0
        currentEncoderCount = self.MOTOR1_A_COUNT
        currentEncoderCount2 = self.MOTOR2_A_COUNT
        finalAmount = 360 * (AMOUNT / 180)
        self.ANGLE_TURNING = True
        self.TURNING_RIGHT = False
        while (((self.MOTOR1_A_COUNT - currentEncoderCount) <= finalAmount) or ((self.MOTOR2_A_COUNT - currentEncoderCount2) <= finalAmount)) and (self.RUN_MOTORS == True):
            self.pauseMotorEvent.wait()
            self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1390)
            self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1390)
        self.ANGLE_TURNING = False
        self.MOTOR_FINISHED = True
        self.motorStop(0)

    def pauseMotor(self, AMOUNT):
        self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1500)
        self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1500)

    def motorStop(self, AMOUNT):
        self.ANGLE_TURNING = False
        self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1500)
        self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1500)

    def forceMotorStop(self, AMOUNT):
        self.pauseMotorEvent.set()
        self.RUN_MOTORS = False
        self.MOTOR_FINISHED = True
        self.ANGLE_TURNING = False
        self.MOTOR_ENCODER_LIDAR2 = 0
        self.pi.set_servo_pulsewidth(self.MOTOR1_GPIO, 1500)
        self.pi.set_servo_pulsewidth(self.MOTOR2_GPIO, 1500)

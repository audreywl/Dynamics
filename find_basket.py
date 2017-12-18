import numpy as np
import cv2
import cv2.aruco as aruco
import time
import pickle
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import wiringpi
import math

def Lazy_Susan(tvecs):
    camera_offset = 3
    tvecs[0] = tvecs[0] + camera_offset
    angle_adjust = np.arctan([tvecs[0,0,0]/tvecs[0,0,2]])
    return angle_adjust

class PWMAdjuster(object):
    def __init__(self):
        self.aimtable = np.genfromtxt('aimtable.csv', delimiter = ',', missing_values='NaN', skip_header=1)
        wiringpi.wiringPiSetupGpio()
        # set #18 to be a PWM output
        wiringpi.pinMode(18, wiringpi.GPIO.PWM_OUTPUT)
        wiringpi.pinMode(13, wiringpi.GPIO.PWM_OUTPUT)
        # set the PWM mode to milliseconds stype
        wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

        # divide down clock
        wiringpi.pwmSetClock(192)
        wiringpi.pwmSetRange(2000)
        self.susan_current_pos = 150
        self.winch_current_pos = 50
        wiringpi.pwmWrite(18, self.susan_current_pos)
        time.sleep(10)
        wiringpi.pwmWrite(13, self.winch_current_pos)
        time.sleep(10)

    def adjust_susan(self, angle):
        """angle is in radians, current_pos in ms"""
        ms = angle*78.8644
        ms = 5* round(ms/5, 0)
        self.susan_current_pos = int(self.susan_current_pos+ms)
        print self.susan_current_pos
        wiringpi.pwmWrite(18, self.susan_current_pos)
        time.sleep(10)


    def adjust_winch(self, dist, height):
        armlen = 17.375
        dist = int(round((dist/armlen)*7, 0))
        height = int(round(height/armlen)*59/4,0)
        angle = self.aimtable[height,dist]
        angle = self.aimtable[0,dist]
        print angle
        angle = math.radians(angle)
        print angle
        ms = angle*(200/.279)
        ms = 5* round(ms/5, 0)
        print ms
        self.winch_current_pos = int(self.winch_current_pos+ms)
        wiringpi.pwmWrite(13, self.winch_current_pos)
        time.sleep(20)


class Detector(object):
    """wrapper to hold all the image processing, AR tag operations"""
    def __init__(self, calibrate=False):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.cap = PiRGBArray(self.camera, size=(640, 480))
        # allow the camera to warmup
        time.sleep(0.1)
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.ar_params = aruco.DetectorParameters_create()
        self.markerLength = 6

        #self.test_marker = aruco.drawMarker(self.dict, 23, 700)
        self.markerPose = [np.full((1,3), np.nan),np.full((1,3), np.nan)]
        if calibrate:
            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self.objp = np.zeros((6*7,3), np.float32)
            self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            self.objpoints = [] # 3d point in real world space
            self.imgpoints = [] # 2d points in image plane.
            self.calibration_params = [None, None, None, None]
        else:
            with open('pookle.p', 'r') as f:
                self.calibration_params = pickle.load(f)

    def update_calibration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.debug_img = gray
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
            self.imgpoints.append(corners2)

            # Draw and display the corners
            self.debug_img = cv2.drawChessboardCorners(gray, (7,6), corners2,ret)
            # cv2.imshow('img',img)
        self.cap.truncate(0)

    def finish_calibration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, self.calibration_params[0], self.calibration_params[1], self.calibration_params[2], self.calibration_params[3] = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None,None)
        with open('pookle.p', 'w') as f:
            pickle.dump(self.calibration_params, f)


    def update(self, frame):
        # transform image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dict, parameters=self.ar_params)

        markers = aruco.drawDetectedMarkers(gray, corners, ids)

        rvecs, tvecs,_ = aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.calibration_params[0], self.calibration_params[1])

        if rvecs is not None and tvecs is not None:
            print rvecs
            print tvecs
            self.debug_img = aruco.drawAxis(markers, self.calibration_params[0], self.calibration_params[1], rvecs, tvecs, 1)
            self.markerPose.append(tvecs)
        else:
            self.debug_img = markers




        # clear the stream in preparation for the next frame
        self.cap.truncate(0)


    def kill_video(self):
        """When everything done, release the capture"""
        self.cap.truncate(0)
        cv2.destroyAllWindows()

def find_pose(detector, debug_video=True):
    for frame in detector.camera.capture_continuous(detector.cap, format="bgr", use_video_port=True):
        image = frame.array
        detector.update(image)
        if debug_video:
            cv2.imshow('debug',detector.debug_img)
            time.sleep(.03)
            cv2.waitKey(30)
        if np.allclose(detector.markerPose[-1], detector.markerPose[-2], atol=.5):
            detector.kill_video()
            return (Lazy_Susan(detector.markerPose[-1]), detector.markerPose[-1][0,0,2], detector.markerPose[-1][0,0,1])


if __name__ == '__main__':
    calibrate = False
    prime_detector = Detector(calibrate=calibrate)
    adjustor = PWMAdjuster()
    if calibrate:
        print 'running calibration'
        for frame in prime_detector.camera.capture_continuous(prime_detector.cap, format="bgr", use_video_port=True):
            image = frame.array
            prime_detector.update_calibration(image)
            cv2.imshow('debug',prime_detector.debug_img)
            print len(prime_detector.imgpoints)
            if len(prime_detector.imgpoints)>4:
                break
            time.sleep(.03)
            cv2.waitKey(30)
        prime_detector.finish_calibration(image)
        print 'calibration complete'
    turn=1
    while abs(turn)>.1:
        turn, distance, height = find_pose(prime_detector)
        print turn
        adjustor.adjust_susan(turn)
    adjustor.adjust_winch(distance, height)
    print 'ready to fire'

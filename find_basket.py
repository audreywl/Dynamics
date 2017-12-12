import numpy as np
import cv2
import cv2.aruco as aruco
import time
from picamera.array import PiRGBArray
from picamera import PiCamera


class Detector(object):
    """wrapper to hold all the image processing, contour finding operations"""
    def __init__(self, calibrate=False, calibration_params=None):
        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.cap = PiRGBArray(camera, size=(640, 480))
        # allow the camera to warmup
        time.sleep(0.1)
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.ar_params = aruco.DetectorParameters_create()
        self.markerLength = 7
        #self.test_marker = aruco.drawMarker(self.dict, 23, 700)
        if calibrate or not calibration_params:
            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self.objp = np.zeros((6*7,3), np.float32)
            self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            self.objpoints = [] # 3d point in real world space
            self.imgpoints = [] # 2d points in image plane.
            self.calibration_params = [None, None, None, None]

    def update_calibration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
            self.imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    def finish_calibration(self):
        _, self.calibration_params[0], self.calibration_params[1], self.calibration_params[2], self.calibration_params[3] = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


    def update(self, frame):
        # transform image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dict, parameters=self.ar_params)

        aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.calibration_params[0], self.calibration_params[1], self.calibration_params[2], self.calibration_params[3])

        self.debug_img = aruco.drawDetectedMarkers(gray, corners, ids)

        # clear the stream in preparation for the next frame
        self.cap.truncate(0)


    def kill_video(self):
        """When everything done, release the capture"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrate = True
    prime_detector = Detector(0, True);
    if calibrate:
        print 'running calibration'
        while len(prime_detector.imgpoints)<5:
            for frame in prime_detector.camera.capture_continuous(prime_detector.cap, format="bgr", use_video_port=True):
                image = frame.array
                prime_detector.update_calibration(image)
                prime_detector.finish_calibration()
        print 'calibration complete'

    for frame in prime_detector.camera.capture_continuous(prime_detector.cap, format="bgr", use_video_port=True):
        image = frame.array
        prime_detector.update(image)
        cv2.imshow('debug',prime_detector.debug_img)
        time.sleep(.03)
        cv2.waitKey(30)

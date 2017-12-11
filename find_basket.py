import numpy as np
import cv2
import cv2.aruco as aruco
import time

class Detector(object):
    """wrapper to hold all the image processing, contour finding operations"""
    def __init__(self, camera):
        self.camera = camera
        self.cap = cv2.VideoCapture(self.camera)
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.ar_params = aruco.DetectorParameters_create()
        #self.test_marker = aruco.drawMarker(self.dict, 23, 700)

    def update(self):
    	# Capture frame-by-frame
        ret, frame = self.cap.read()


        # transform image to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dict, parameters=self.ar_params)

        aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

        self.debug_img = aruco.drawDetectedMarkers(gray, corners, ids)


    def kill_video(self):
        """When everything done, release the capture"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    prime_detector = Detector(0);
    #cv2.imshow('debug', prime_detector.test_marker)
    running = True
    while  running:
        prime_detector.update()
        cv2.imshow('debug',prime_detector.debug_img)
        time.sleep(.03)
        cv2.waitKey(30)

import numpy as np
import cv2
import cv2.aruco as aruco
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
 



class Detector(object):
    """wrapper to hold all the image processing, contour finding operations"""
    def __init__(self, camera):
	# initialize the camera and grab a reference to the raw camera capture
	self.camera = PiCamera()
	self.camera.resolution = (640, 480)
	self.camera.framerate = 32
	self.cap = PiRGBArray(camera, size=(640, 480))
	# allow the camera to warmup
	time.sleep(0.1)
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.ar_params = aruco.DetectorParameters_create()
        #self.test_marker = aruco.drawMarker(self.dict, 23, 700)

    def update(self, frame):
        # transform image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dict, parameters=self.ar_params)

        #aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

        self.debug_img = aruco.drawDetectedMarkers(gray, corners, ids)
	
	# clear the stream in preparation for the next frame
	self.cap.truncate(0)


    def kill_video(self):
        """When everything done, release the capture"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    prime_detector = Detector(0);
    for frame in prime_detector.camera.capture_continuous(prime_detector.cap, format="bgr", use_video_port=True):
	image = frame.array
        prime_detector.update(image)
        cv2.imshow('debug',prime_detector.debug_img)
        time.sleep(.03)
        cv2.waitKey(30)


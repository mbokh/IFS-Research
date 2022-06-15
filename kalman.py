import cv2
import numpy as np


class KalmanFilter:
	def __init__(self, boundingBoxData):
		self.kalman = cv2.KalmanFilter(8, 4)
		self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
												  [0,1,0,0,0,0,0,0],
												  [0,0,0,0,1,0,0,0],
												  [0,0,0,0,0,1,0,0]],np.float32)

		self.kalman.transitionMatrix = np.array([[1,0,1,0,0,0,0,0],
												 [0,1,0,1,0,0,0,0],
												 [0,0,1,0,0,0,0,0],
												 [0,0,0,1,0,0,0,0],
												 [0,0,0,0,1,0,1,0],
												 [0,0,0,0,0,1,0,1],
												 [0,0,0,0,0,0,1,0],
												 [0,0,0,0,0,0,0,1]],np.float32)

		self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1
		'''np.array([[0.02,0,0,0,0,0,0,0],
												[0,0.02,0,0,0,0,0,0],
												[0,0,0.02,0,0,0,0,0],
												[0,0,0,0.02,0,0,0,0],
												[0,0,0,0,0.02,0,0,0],
												[0,0,0,0,0,0.02,0,0],
												[0,0,0,0,0,0,0.02,0],
												[0,0,0,0,0,0,0,0.02]],np.float32)'''

		self.projectionMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
										  [0, 1, 0, 0, 0, 0, 0, 0],
										  [0, 0, 1, 0, 0, 0, 0, 0],
										  [0, 0, 0, 1, 0, 0, 0, 0],
										  [0, 0, 0, 0, 1, 0, 0, 0],
										  [0, 0, 0, 0, 0, 1, 0, 0]], np.float32)


		self.kalman.statePost = np.array([[boundingBoxData[4]],
				     [boundingBoxData[5]],
					 [0],
					 [0],
				     [boundingBoxData[2]],
				     [boundingBoxData[3]],
					  [0],
					  [0]], np.float32)
		self.getPrediction()

	def getPrediction(self):
		return np.matmul(self.projectionMatrix, self.kalman.predict())

	def update(self, boundingBoxData):
		m = np.array([[boundingBoxData[4]],
				     [boundingBoxData[5]],
				     [boundingBoxData[2]],
				     [boundingBoxData[3]]], np.float32)
		self.kalman.correct(m)

	def updateFromPrediction(self):
		self.kalman.correct(np.matmul(self.kalman.measurementMatrix, self.kalman.predict()))
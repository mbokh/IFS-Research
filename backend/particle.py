from backend import kalman
import numpy as np


def roundBBoxCentroid(bBox):
	bList = list(bBox)
	bList[4] = round(bBox[4], 1)  # Round centroid coordinates for sake of space
	bList[5] = round(bBox[5], 1)
	return bList


class Particle:
	def __init__(self, particleId, boundingBox, brightness, frameNum):
		self.pId = particleId
		self.frameNumAppeared = frameNum

		self.kalmanFilter = kalman.KalmanFilter(boundingBox)
		self.particleData = [(roundBBoxCentroid(boundingBox), brightness, 0)]
		self.spectraData = []

	def getKalmanPrediction(self):
		return self.kalmanFilter.getPrediction()

	def getPreviousOcclusionCount(self):
		return self.particleData[-1][2]

	def getPreviousBrightness(self):
		return self.particleData[-1][1]

	def getPreviousBoundingBox(self):
		return self.particleData[-1][0]

	def propagateFromPrediction(self):
		bBoxPredicted = self.kalmanFilter.updateFromPrediction()
		self.particleData.append((roundBBoxCentroid(bBoxPredicted), self.getPreviousBrightness(), self.getPreviousOcclusionCount() + 1))

	def updateBBox(self, bBox, brightness): #(x, y, w, h, cX, cY)
		self.kalmanFilter.update(bBox)
		self.particleData.append((roundBBoxCentroid(bBox), brightness, 0))

	def addSpectraData(self, data):  #temp, spectra, code
		#Flip spectra for final logging since they are backward during processing
		self.spectraData.append((data[0], np.flip(data[1]).astype(np.uint64).tolist(), data[2]))  #Want spectra as simple python list so that it can be pickled
		assert(len(self.spectraData) == len(self.particleData)) #Also values are ~10^11+, round to int to save space in pickle and csv file

	def prepareForPickling(self):
		self.kalmanFilter = None #Can't pickle kalman, and we don't need it anyway, so simply destroy it
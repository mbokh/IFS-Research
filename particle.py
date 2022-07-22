import kalman


class Particle:
	def __init__(self, particleId, boundingBox, brightness, frameNum):
		self.pId = particleId
		self.frameNumAppeared = frameNum

		self.kalmanFilter = kalman.KalmanFilter(boundingBox)
		self.particleData = [(boundingBox, brightness, 0)]
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
		self.particleData.append((bBoxPredicted, self.getPreviousBrightness(), self.getPreviousOcclusionCount() + 1))

	def updateBBox(self, bBox, brightness):
		self.kalmanFilter.update(bBox)
		self.particleData.append((bBox, brightness, 0))

	def addSpectraData(self, data):  #temp, spectra, code
		self.spectraData.append(data)
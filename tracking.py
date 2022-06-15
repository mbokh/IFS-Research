import math
from scipy.optimize import linear_sum_assignment
import cv2
import kalman
import numpy as np


def detectObjects(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	blured = cv2.GaussianBlur(gray, (11, 11), 0, 0)
	sharpened = cv2.addWeighted(gray, 5, blured, -4, 0)

	cv2.rectangle(sharpened, (0, 0), (896 - 200, 448), 0, -1) #Block the spectra

	#threshold, dummyImage = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	binary = cv2.threshold(sharpened, 40, 255, cv2.THRESH_BINARY)[1]

	numRegions, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

	data = []
	for i in range(numRegions):
		if i == 0: #Skip background
			continue
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		(cX, cY) = centroids[i]

		if w == 1 and h == 1:
			continue
		data.append((x, y, w, h, cX, cY))

	return data

def distance(v):
	return math.sqrt(v[0]**2 + v[1]**2)

def calculateCost(previousPrediction, currentBBox, previousBBox, occlusion):
	toTarget = (currentBBox[4] - previousPrediction[0, 0], currentBBox[5] - previousPrediction[1, 0])

	dist = distance(toTarget)
	size = (previousPrediction[4, 0] - currentBBox[2])**2 + (previousPrediction[5, 0] - currentBBox[3])**2

	velocityVector = (previousPrediction[2][0], previousPrediction[3][0])
	displacement = (currentBBox[4] - previousBBox[4], currentBBox[5] - previousBBox[5])
	occlusionScore = (occlusion**3) + 20

	if distance(displacement) == 0 or distance(velocityVector) < 0.25:
		return dist + (6 * size) + occlusionScore # Can't use angle metric here

	cosine = ((velocityVector[0] * displacement[0]) + (velocityVector[1] * displacement[1])) / (distance(displacement) * distance(velocityVector))
	angleScore = 10 * math.exp(-2 * cosine)
	#if dist > 3 * distance(velocityVector):
	#	return 10000
	return dist + (6 * size) + angleScore + occlusionScore

class MultiObjectTracker:
	def __init__(self):
		self.applyHungarian = False
		self.previousFrameData = []
		self.previouslyUsedId = -1

	def getNewId(self):
		self.previouslyUsedId += 1
		return self.previouslyUsedId

	def processImage(self, img):
		boundingBoxes = detectObjects(img)

		if not self.applyHungarian and len(boundingBoxes) != 0: #First time seeing particles, no Hungarian alg needed
			self.applyHungarian = True
			for b in boundingBoxes:
				self.previousFrameData.append([self.getNewId(), kalman.KalmanFilter(b), b, 0])
		else: #Not the first frame with particle, need Hungarian
			size = max(len(self.previousFrameData), len(boundingBoxes))
			costMatrix = np.zeros((size, size))
			for i in range(size):
				predictedState = None if i >= len(self.previousFrameData) else self.previousFrameData[i][1].getPrediction()
				for j in range(size):
					if i >= len(self.previousFrameData) or j >= len(boundingBoxes):
						costMatrix[i, j] = 0
					else:
						costMatrix[i, j] = calculateCost(predictedState, boundingBoxes[j], self.previousFrameData[i][2], self.previousFrameData[i][3])

			rows, cols = linear_sum_assignment(costMatrix)

			cost = costMatrix[rows, cols].sum()
			while True:
				size += 1
				newCostMatrix = np.zeros((size, size))
				newCostMatrix[0:(size - 1), 0:(size - 1)] = costMatrix
				costMatrix = newCostMatrix
				for i in range(size):
					costMatrix[0, i] = 0
					costMatrix[i, 0] = 0
				newRows, newCols = linear_sum_assignment(costMatrix)
				newCost = costMatrix[newRows, newCols].sum()
				if newCost < cost:
					cost = newCost
					rows = newRows
					cols = newCols
				break

			toBeDeleted = []
			for i in range(len(rows)):
				if rows[i] >= len(self.previousFrameData): #Particle appears
					b = boundingBoxes[cols[i]]
					self.previousFrameData.append([self.getNewId(), kalman.KalmanFilter(b), b, 0])
				elif cols[i] >= len(boundingBoxes): #Particle dissappears
					self.previousFrameData[rows[i]][3] += 1
					if self.previousFrameData[rows[i]][3] == 6:
						toBeDeleted.append(rows[i])
					else:
						self.previousFrameData[rows[i]][1].updateFromPrediction()

				else: #Assignment
					self.previousFrameData[rows[i]][2] = boundingBoxes[cols[i]]
					self.previousFrameData[rows[i]][1].update(boundingBoxes[cols[i]])
					self.previousFrameData[rows[i]][3] = 0 #Reset occlusion count

			toBeDeleted.sort(reverse=True)
			for i in toBeDeleted:
				del self.previousFrameData[i]

	def getPreviouslyUsedId(self):
		return self.previouslyUsedId

	def getTrackingData(self):
		data = []
		for particleId, kf, box, occlusionCount in self.previousFrameData:
			if occlusionCount == 0:
				data.append((particleId, box))
		return data
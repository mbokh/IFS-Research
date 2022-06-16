import math
from scipy.optimize import linear_sum_assignment
import cv2
import kalman
import numpy as np

LARGE_WEIGHT = 100000

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
	distanceCost = 100 if dist >= 30 else 0.1 * dist * dist

	
	size = (abs(previousPrediction[4, 0]) - abs(currentBBox[2]))**2 + (abs(previousPrediction[5, 0]) - abs(currentBBox[3]))**2
	sizeCost = 100 if size >= 25 else 0.07 * size * size


	velocityVector = (previousPrediction[2][0], previousPrediction[3][0])
	displacement = (currentBBox[4] - previousBBox[4], currentBBox[5] - previousBBox[5])

	occlusionScore = (occlusion**3) + 10

	if distance(displacement) == 0 or distance(velocityVector) < 0.25:
		return (1*distanceCost) + (1 * sizeCost) + occlusionScore # Can't use angle metric here

	cosine = ((velocityVector[0] * displacement[0]) + (velocityVector[1] * displacement[1])) / (distance(displacement) * distance(velocityVector))
	angleScore = 10 * math.exp(-2 * cosine)
	#if dist > 3 * distance(velocityVector):
	#	return 10000
	return (1*distanceCost) + (1 * sizeCost)  + occlusionScore


def appendZeros(costMatrix):
	oldSize = costMatrix.shape[0]
	newCostMatrix = np.zeros((oldSize + 1, oldSize + 1))
	newCostMatrix[0:oldSize, 0:oldSize] = costMatrix
	for i in range(oldSize + 1):
		newCostMatrix[oldSize, i] = 0
		newCostMatrix[i, oldSize] = 0
	return newCostMatrix

def hasLargeCost(costMatrix, r, c):
	for i in range(len(r)):
		if costMatrix[r[i], c[i]] > 100:
			return True
	return False

def iterativelyFindBetterSolutions(rows, cols, costMatrix, oldCost):
	if not hasLargeCost(costMatrix, rows, cols):
		return rows, cols, costMatrix

	while True:
		newCostMatrix = appendZeros(costMatrix)
		newRows, newCols = linear_sum_assignment(newCostMatrix)
		newCost = newCostMatrix[newRows, newCols].sum()

		if newCost >= oldCost:
			return rows, cols, costMatrix

		if not hasLargeCost(newCostMatrix, newRows, newCols):
			return newRows, newCols, newCostMatrix

		oldCost = newCost
		rows = newRows
		cols = newCols
		costMatrix = newCostMatrix

		length = newCostMatrix.shape[0]
		for i in range(length):
			newCostMatrix[length - 1, i] = LARGE_WEIGHT
			newCostMatrix[i, length - 1] = LARGE_WEIGHT

		index = 0
		for v in range(len(rows)):
			if rows[v] == length - 1:
				index = v
				break
		newCostMatrix[length - 1, cols[index]] = 0
		for v in range(len(cols)):
			if cols[v] == length - 1:
				index = v
				break
		newCostMatrix[rows[index], length - 1] = 0


class MultiObjectTracker:
	def __init__(self):
		self.applyHungarian = False
		self.previousFrameData = []
		self.previouslyUsedId = -1

	def getNewId(self):
		self.previouslyUsedId += 1
		return self.previouslyUsedId

	def calculateInitialCostMatrix(self, boundingBoxes):
		size = max(len(self.previousFrameData), len(boundingBoxes))
		costMatrix = np.zeros((size, size))
		for i in range(size):
			predictedState = None if i >= len(self.previousFrameData) else self.previousFrameData[i][1].getPrediction()
			for j in range(size):
				if i >= len(self.previousFrameData) or j >= len(boundingBoxes):
					costMatrix[i, j] = 0
				else:
					costMatrix[i, j] = calculateCost(predictedState, boundingBoxes[j], self.previousFrameData[i][2], self.previousFrameData[i][3])
		return size, costMatrix

	def processImage(self, img):
		boundingBoxes = detectObjects(img)

		if not self.applyHungarian and len(boundingBoxes) != 0: #First time seeing particles, no Hungarian alg needed
			self.applyHungarian = True
			for b in boundingBoxes:
				self.previousFrameData.append([self.getNewId(), kalman.KalmanFilter(b), b, 0])
			return
		#Not the first frame with particle, now need Hungarian
		size, costMatrix = self.calculateInitialCostMatrix(boundingBoxes)
		rows, cols = linear_sum_assignment(costMatrix)

		if size > 0:
			rows, cols, costMatrix = iterativelyFindBetterSolutions(rows, cols, costMatrix, costMatrix[rows, cols].sum())
		#print(costMatrix)

		toBeDeleted = []
		for i in range(len(rows)):
			if rows[i] >= len(self.previousFrameData): #Particle appears
				b = boundingBoxes[cols[i]]
				self.previousFrameData.append([self.getNewId(), kalman.KalmanFilter(b), b, 0])
			elif cols[i] >= len(boundingBoxes): #Particle dissappears
				self.previousFrameData[rows[i]][3] += 1
				if self.previousFrameData[rows[i]][3] == 4:
					toBeDeleted.append(rows[i])
				else:
					bBoxPredicted = self.previousFrameData[rows[i]][1].updateFromPrediction()
					self.previousFrameData[rows[i]][2] = bBoxPredicted

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
			#if occlusionCount == 0:
			data.append((particleId, box, occlusionCount))
		return data
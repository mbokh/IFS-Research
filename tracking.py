import math
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import database

LARGE_WEIGHT = 100000

def detectObjects(img, videoSource):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	medianBlured = cv2.medianBlur(gray, 3)

	blured = cv2.GaussianBlur(medianBlured, (11, 11), 0, 0)
	sharpened = cv2.addWeighted(medianBlured, 5, blured, -4, 0)

	spectraStart, spectraEnd = videoSource.getSpectraPartition()
	cv2.rectangle(sharpened, (spectraStart, 0), (spectraEnd, videoSource.getHeight()), 0, -1)  # Block the spectra

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

		if w * h < 2:
			continue
		data.append((x, y, w, h, cX, cY))

	return data, gray

def distance(v):
	return math.sqrt(v[0]**2 + v[1]**2)


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

		length = newCostMatrix.shape[0] #Set newly added col/rows to large weight
		for i in range(length):
			newCostMatrix[length - 1, i] = LARGE_WEIGHT
			newCostMatrix[i, length - 1] = LARGE_WEIGHT

		index = 0  #For the specific assignments made, make them 0 weight
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

def getBrightnessAroundPoint(img, x, y):
	v = int(img[boundInt(y + 1, 0, img.shape[0]), boundInt(x - 1, 0, img.shape[1])]) + \
		int(img[boundInt(y + 1, 0, img.shape[0]), boundInt(x, 0, img.shape[1])]) + \
		int(img[boundInt(y + 1, 0, img.shape[0]), boundInt(x + 1, 0, img.shape[1])]) + \
		int(img[boundInt(y, 0, img.shape[0]), boundInt(x - 1, 0, img.shape[1])]) + \
		int(img[boundInt(y, 0, img.shape[0]), boundInt(x, 0, img.shape[1])]) + \
		int(img[boundInt(y, 0, img.shape[0]), boundInt(x + 1, 0, img.shape[1])]) + \
		int(img[boundInt(y - 1, 0, img.shape[0]), boundInt(x - 1, 0, img.shape[1])]) + \
		int(img[boundInt(y - 1, 0, img.shape[0]), boundInt(x, 0, img.shape[1])]) + \
		int(img[boundInt(y - 1, 0, img.shape[0]), boundInt(x + 1, 0, img.shape[1])])
	return int(v / 9)

def boundInt(v, minV, maxV): #Max is exclusive
	return int(min(max(v, minV), maxV - 1))


def calculateCost(previousPrediction, currentBBox, particle, newGray):
	toTarget = (currentBBox[4] - previousPrediction[0, 0], currentBBox[5] - previousPrediction[1, 0])
	dist = distance(toTarget)
	distanceCost = 100 if dist >= 30 else 0.1 * dist * dist

	size = (abs(previousPrediction[4, 0]) - abs(currentBBox[2])) ** 2 + (abs(previousPrediction[5, 0]) - abs(currentBBox[3])) ** 2
	sizeCost = 100 if size >= 25 else 0.07 * size * size


	brightnessDelta = abs(particle.getPreviousBrightness() - getBrightnessAroundPoint(newGray, currentBBox[4], currentBBox[5]))
	brightnessCost = 100 if brightnessDelta >= 50 else 0.02 * brightnessDelta * brightnessDelta


	#velocityVector = (previousPrediction[2][0], previousPrediction[3][0])
	#displacement = (currentBBox[4] - previousBBox[4], currentBBox[5] - previousBBox[5])

	occlusionScore = (particle.getPreviousOcclusionCount() ** 3) + 10

	#if distance(displacement) == 0 or distance(velocityVector) < 0.25:
	#	return (1 * distanceCost) + (1 * sizeCost) + occlusionScore + brightnessCost # Can't use angle metric here

	#cosine = ((velocityVector[0] * displacement[0]) + (velocityVector[1] * displacement[1])) / (distance(displacement) * distance(velocityVector))
	#angleScore = 10 * math.exp(-2 * cosine)
	# if dist > 3 * distance(velocityVector):
	#	return 10000
	return (1 * distanceCost) + (1 * sizeCost) + occlusionScore + brightnessCost


def calculateInitialCostMatrix(previousIds, boundingBoxes, newGray):
	size = max(len(previousIds), len(boundingBoxes))
	costMatrix = np.zeros((size, size))
	for i in range(size):
		predictedState = None if i >= len(previousIds) else database.getParticleById(previousIds[i]).getKalmanPrediction()
		for j in range(size):
			if i >= len(previousIds) or j >= len(boundingBoxes):
				costMatrix[i, j] = 0
			else:
				costMatrix[i, j] = calculateCost(predictedState, boundingBoxes[j], database.getParticleById(previousIds[i]), newGray)

	for i in range(size):
		for j in range(size):
			if 0 < costMatrix[i, j] < 15: #If the cost is so small, it must be a good fit, so make sure it gets chosen by making all
				for k in range(size): #other weights very large
					if costMatrix[i, k] > 0 and k != j:
						costMatrix[i, k] = LARGE_WEIGHT
				for k in range(size):
					if costMatrix[k, j] > 0 and k != i:
						costMatrix[k, j] = LARGE_WEIGHT
	#print(costMatrix)
	return size, costMatrix


class MultiObjectTracker:
	def __init__(self):
		self.applyHungarian = False

	def processImage(self, img, frameNum, video):
		boundingBoxes, grayscale = detectObjects(img, video)

		if not self.applyHungarian and len(boundingBoxes) != 0: #First time seeing particles, no Hungarian alg needed
			self.applyHungarian = True
			for b in boundingBoxes:
				database.createNewParticle(b, getBrightnessAroundPoint(grayscale, b[4], b[5]), frameNum)
			return
		#Not the first frame with particle, now need Hungarian
		previousIds = database.getIdsOfPreviousFrame()
		size, costMatrix = calculateInitialCostMatrix(previousIds, boundingBoxes, grayscale)
		rows, cols = linear_sum_assignment(costMatrix)

		if size > 0:
			rows, cols, costMatrix = iterativelyFindBetterSolutions(rows, cols, costMatrix, costMatrix[rows, cols].sum())
		#print(costMatrix)

		for i in range(len(rows)):
			if rows[i] >= len(previousIds): #Particle appears
				b = boundingBoxes[cols[i]]
				database.createNewParticle(b, getBrightnessAroundPoint(grayscale, b[4], b[5]), frameNum)
			elif cols[i] >= len(boundingBoxes): #Particle dissappears
				database.isOccluded(previousIds[rows[i]])
			else: #Assignment
				b = boundingBoxes[cols[i]]
				database.assignNewData(previousIds[rows[i]], b, getBrightnessAroundPoint(grayscale, b[4], b[5]))
import math
from scipy.optimize import linear_sum_assignment
import cv2
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

		data.append((x, y, w, h, cX, cY))

	return data


def calculateWeight(previous, current):
	return math.sqrt((previous[0] - current[0])**2 + (previous[1] - current[1])**2)


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

		if not self.applyHungarian: #Haven't seen particles yet
			if len(boundingBoxes) == 0:
				return []
			else: #First time seeing particles, no Hungarian alg needed
				self.applyHungarian = True
				for b in boundingBoxes:
					self.previousFrameData.append((self.getNewId(), b))
				return self.previousFrameData
		else: #Not the first frame with particle, need Hungarian
			size = max(len(self.previousFrameData), len(boundingBoxes))
			costMatrix = np.zeros((size, size))
			for i in range(size):
				for j in range(size):
					if i >= len(self.previousFrameData) or j >= len(boundingBoxes):
						costMatrix[i, j] = 0
					else:
						costMatrix[i, j] = calculateWeight(self.previousFrameData[i][1], boundingBoxes[j])

			rows, cols = linear_sum_assignment(costMatrix)
			newData = []
			for i in range(len(rows)):
				if rows[i] >= len(self.previousFrameData): #Particle appears
					newData.append((self.getNewId(), boundingBoxes[cols[i]]))
				elif cols[i] >= len(boundingBoxes): #Particle dissappears
					continue
				else: #Assignment
					#oldData = self.previousFrameData[rows[i]]
					#newData = boundingBoxes[cols[i]]
					newData.append((self.previousFrameData[rows[i]][0], boundingBoxes[cols[i]]))
			self.previousFrameData = newData
			return self.previousFrameData

	def getPreviouslyUsedId(self):
		return self.previouslyUsedId

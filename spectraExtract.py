import numpy as np
import cv2
import math

import videoSource

spectraLineAngle = 2.2 #Degrees, positive because in image plane, y axis is "down"
mappingTransform = np.array([[1, 0, -349],
							 [0, 1, -5],
							 [0, 0, 1]])
pointsRotationTransform = np.array([[math.cos(-spectraLineAngle * (math.pi / 180.0)), -math.sin(-spectraLineAngle * (math.pi / 180.0)), 0],
							  [math.sin(-spectraLineAngle * (math.pi / 180.0)), math.cos(-spectraLineAngle * (math.pi / 180.0)), 0],
							  [0, 0, 1]])
padding = 50
paddingTransform = np.array([[1, 0, 0],
							 [0, 1, padding],
							 [0, 0, 1]])

extendLeft = 150
extendRight = 150

def mapParticleToSpectra(x, y):
	coords = np.array([x, y, 1])
	m = np.matmul(mappingTransform, coords)
	m = np.matmul(paddingTransform, m)
	m = np.matmul(pointsRotationTransform, m)
	return int(m[0]), int(m[1])

def addBlackPadding(f):
	newFrame = np.zeros((videoSource.getHeight() + padding, videoSource.getWidth() + padding, 3), np.uint8)
	newFrame[padding:, :-padding] = f[:, :]
	return newFrame

def boundingBoxesOverlap(r1, r2):
	x1, y1, w1, h1 = r1
	x2, y2, w2, h2 = r2
	return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

def subtractRange(initialSet, operand):
	processed = []
	for initial in initialSet:
		if initial[0] >= operand[0] and initial[1] > operand[1]:
			processed.append((operand[1] + 1, initial[1]))
		elif initial[0] >= operand[0] and initial[1] <= operand[1]:
			continue #Entirely removed
		elif initial[0] < operand[0] and initial[1] > operand[1]:
			processed.append((initial[0], operand[1] - 1)) #Break into 2 segments
			processed.append((operand[1] + 1, initial[1]))
		else:
			processed.append((initial[0], operand[1] - 1))
	return processed


def extractRawSpectra(frame, trackingData):
	padded = addBlackPadding(frame)

	height, width = padded.shape[:2]
	rotate_matrix = cv2.getRotationMatrix2D(center=(0, padding), angle=spectraLineAngle, scale=1)
	rotated_image = cv2.warpAffine(src=padded, M=rotate_matrix, dsize=(width, height))


	gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

	bBoxes = []
	for particleId, box, occlusionCount in trackingData:
		x1, y1, w1, h1, cX1, cY1 = box
		mappedCX, mappedCY = mapParticleToSpectra(cX1, cY1)
		#Length preserved
		cv2.rectangle(rotated_image, (mappedCX - extendLeft, mappedCY + int(h1 / 2)), (mappedCX + extendLeft, mappedCY - int(h1 / 2)), 255, 1)
		bBoxes.append((particleId, (mappedCX - extendLeft, mappedCY - int(h1 / 2), mappedCX + extendRight, mappedCY + int(h1 / 2))))

	#Contains index of bBox

	measurements = []
	for i in range(len(bBoxes)):
		conflicts = []
		minX = bBoxes[i][1][0]
		maxX = bBoxes[i][1][2]
		for j in range(len(bBoxes)):
			if j == i:
				continue
			if boundingBoxesOverlap(bBoxes[i][1], bBoxes[j][1]):
				conflicts.append(j)
				minX = min(minX, bBoxes[i][1][0])
				maxX = max(maxX, bBoxes[i][1][2])

		initialRange = [(bBoxes[i][1][1], bBoxes[i][1][3])]
		if len(conflicts) == 0: #No conflit case
			measurements.append((bBoxes[i][0], 0, integrateOverRanges(gray, bBoxes[i][1][0], bBoxes[i][1][2], initialRange)))
		else:
			for k in conflicts:
				initialRange = subtractRange(initialRange, (bBoxes[k][1][1], bBoxes[k][1][3]))
			largeBands = []
			for segment in initialRange:
				if segment[1] - segment[0] > 1: #Ignore bands with height 1
					largeBands.append(segment)

			if len(largeBands) > 0:
				measurements.append((bBoxes[i][0], 1, integrateOverRanges(gray, bBoxes[i][1][0], bBoxes[i][1][2], largeBands)))
			else:
				measurements.append((bBoxes[i][0], 2, integrateOverRanges(gray, minX, maxX, [(bBoxes[i][1][1], bBoxes[i][1][3])])))

	return rotated_image, measurements


def integrateOverRanges(frame, startX, endX, yRange):
	vector = np.zeros(endX - startX + 1, np.uint8)
	height = 0
	for startY, endY in yRange:
		vector = vector + frame[startY:endY + 1, startX:endX + 1].sum(axis=0)
		height += endY - startY + 1
	return vector / height
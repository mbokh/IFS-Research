import numpy as np
import cv2
import math
import time

import videoSource
import colorID
import database
import calibration.calib as calibration
import calibration.conflictResolution as conflictResolution
import calibration.conversion as conversion

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

extendLeft = calibration.getExtendsLeft()
extendRight = calibration.getExtendsRight() - 1

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
	x1, y1, r1, b1 = r1
	x2, y2, r2, b2 = r2
	return x1 < r2 and r1 > x2 and y1 < b2 and b1 > y2

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
		elif initial[0] < operand[0] and initial[1] <= operand[1]:
			processed.append((initial[0], operand[1] - 1))
		else: #In case segments don't overlap at all, need this case
			processed.append((initial[0], initial[1]))
	return processed


def extractRawSpectra(frame):
	t = time.time_ns()
	origHeight, origWidth = frame.shape[:2]
	padded = addBlackPadding(frame)

	height, width = padded.shape[:2]
	rotate_matrix = cv2.getRotationMatrix2D(center=(0, padding), angle=spectraLineAngle, scale=1)

	gray = cv2.cvtColor(cv2.warpAffine(src=padded, M=rotate_matrix, dsize=(width, height)), cv2.COLOR_BGR2GRAY)

	cv2.line(padded, (0, padding), (origWidth - 1, padding), (255, 255, 255), 1)
	cv2.line(padded, (0, origHeight - 1 + padding), (origWidth - 1, origHeight - 1 + padding), (255, 255, 255), 1)
	cv2.line(padded, (0, padding), (0, origHeight - 1 + padding), (255, 255, 255), 1)
	cv2.line(padded, (origWidth - 1, padding), (origWidth - 1, origHeight - 1 + padding), (255, 255, 255), 1)
	debugImage = cv2.warpAffine(src=padded, M=rotate_matrix, dsize=(width, height))

	particles = dict() #Contain pId -> ((left, top, right, bottom), particleWidth)
	for particleId, box, occlusionCount in database.getLastBoundingBoxes():
		x1, y1, w1, h1, cX1, cY1 = box
		mappedCX, mappedCY = mapParticleToSpectra(cX1, cY1)
		#Length preserved
		cv2.rectangle(debugImage, (mappedCX - extendLeft, mappedCY + int(h1 / 2)), (mappedCX + extendLeft, mappedCY - int(h1 / 2)), colorID.getColorOfId(particleId), 1)
		cv2.putText(debugImage, str(particleId), (mappedCX - extendLeft - 15, mappedCY - int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if occlusionCount == 0 else (0, 0, 255), 1, cv2.LINE_AA, False)
		particles[particleId] = ((mappedCX - extendLeft, mappedCY - int(h1 / 2), mappedCX + extendRight, mappedCY + int(h1 / 2)), w1)


	resolvableConflicts = dict() #Includes no and partial conflicts, by id
	unresolvableConflicts = dict() #Full conflicts
	for i in particles:
		conflicts = set()  # Contains pId
		for j in particles:
			if j == i:
				continue
			if boundingBoxesOverlap(particles[i][0], particles[j][0]):
				conflicts.add(j)
		data, code = getConflictType(i, particles, conflicts)
		if code == 0 or code == 1:
			resolvableConflicts[i] = (averageOverRanges(gray, particles[i][0][0], particles[i][0][2], data, particles[i][1]), code)
		else:
			unresolvableConflicts[i] = (data, gray.copy()) # A little expensive to copy, but worth the trouble
	#Try to subtract away known spectra
	oldLength = len(unresolvableConflicts)
	while len(unresolvableConflicts) > 0:
		for conflictId in list(unresolvableConflicts.keys()):
			conflicts, imageSlice = unresolvableConflicts[conflictId]
			for c in list(conflicts): #Remove conflicts when spectra are known
				if c in resolvableConflicts:
					xRange, yRange, spectraSegment = getSubtractionBounds(particles[conflictId][0], particles[c][0], resolvableConflicts[c][0]) #Subtraction is broadcasted to each row
					imageSlice[yRange[0]:(yRange[1] + 1), xRange[0]:(xRange[1] + 1)] = imageSlice[yRange[0]:(yRange[1] + 1), xRange[0]:(xRange[1] + 1)] - spectraSegment #Can't use -= with numpy slice
					conflicts.remove(c)
			data, code = getConflictType(conflictId, particles, conflicts)
			if code == 0 or code == 1:
				resolvableConflicts[conflictId] = (averageOverRanges(imageSlice, particles[conflictId][0][0], particles[conflictId][0][2], data, particles[conflictId][1]), code + 2)
				del unresolvableConflicts[conflictId]

		if len(unresolvableConflicts) == oldLength: #No progress made, nothing further can be subtracted
			break
		oldLength = len(unresolvableConflicts)

	finalSpectra = dict() #pId -> temp, spectra, code
	#Now, have to do least-squares or brute for demixing
	while len(unresolvableConflicts) > 0:
		conflictId = list(unresolvableConflicts.keys())[0]
		conflicts, imageSlice = unresolvableConflicts[conflictId]
		minBound = particles[conflictId][0][0] #Incorporates original particle
		maxBound = particles[conflictId][0][2]
		offsets = {conflictId : 0}
		for c in conflicts:
			minBound = min(minBound, particles[c][0][0])
			maxBound = max(maxBound, particles[c][0][2])
			offsets[c] = particles[c][0][0] - particles[conflictId][0][0]

		spectra = averageOverRanges(imageSlice, minBound, maxBound, [(particles[conflictId][0][1], particles[conflictId][0][3])], particles[conflictId][1])

		resultDict = conflictResolution.resolve(spectra, offsets) #pId -> temp, spectra
		for pId in resultDict:
			finalSpectra[pId] = (resultDict[pId][0], resultDict[pId][1], 4)
			if conflictId in unresolvableConflicts:
				del unresolvableConflicts[pId]

	#Add from data from resolvable conflicts
	for pId in resolvableConflicts:
		temp = conflictResolution.resolve(resolvableConflicts[pId][0], {pId: 0})
		originalSpectra = conversion.convertPixelToPhysical(resolvableConflicts[pId][0])
		finalSpectra[pId] = (temp[pId][0], originalSpectra, resolvableConflicts[pId][1])

	#Log in database
	database.addSpectraData(finalSpectra)
	print((time.time_ns() - t) / 1000000000)
	return debugImage, finalSpectra


def getSubtractionBounds(baseBox, conflictBox, conflictSpectra): #Particle positions
	x1, y1, r1, b1 = baseBox
	x2, y2, r2, b2 = conflictBox

	assert(r1 - x1 == r2 - x2)
	xRange = [x1, r2] #For case of conflict box to left of main box or on top of box
	spectraSegment = conflictSpectra[x1 - x2:]
	if x1 < x2:
		xRange = [x2, r1]
		spectraSegment = conflictSpectra[:(r2 - x2) - (x2 - x1) + 1]

	yRange = [y1, b1] #Assumes case where conflict box is taller in both directions vertically
	if y2 > y1:
		yRange = [y2, yRange[1]]
	if b1 > b2:
		yRange = [yRange[0], b2]
	return xRange, yRange, spectraSegment

def getConflictType(pId, particles, conflicts):
	initialRange = [(particles[pId][0][1], particles[pId][0][3])]  # Vertical range
	if len(conflicts) == 0:  # No conflict case
		return initialRange, 0
	else:
		for k in conflicts:
			initialRange = subtractRange(initialRange, (particles[k][0][1], particles[k][0][3]))
		largeBands = []
		for segment in initialRange:
			if segment[1] - segment[0] > 1:  # Ignore bands with height 1
				largeBands.append(segment)

		if len(largeBands) > 0:
			return largeBands, 1
		else:
			return conflicts, 2


def averageOverRanges(frame, startX, endX, yRange, particleWidth):
	vector = np.zeros(endX - startX + 1, np.uint8)
	height = 0
	for startY, endY in yRange:
		vector = vector + frame[startY:endY + 1, startX:endX + 1].sum(axis=0)
		height += endY - startY + 1
	return vector / (height * max(1, particleWidth)) #Particle width can be 0
	#In case of occluded particle, kalman could predict 0 width
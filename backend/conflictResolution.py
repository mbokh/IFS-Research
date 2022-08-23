import numpy as np
from backend import calib as calibration, conversion
from scipy.optimize import least_squares
import sys
import matplotlib.pyplot as plt

tempGranularity = 12

bestLoss = sys.float_info.max
bestParams = []

def normalizeOffsets(offsets):
	minValue = min(offsets)
	offsets = [i - minValue for i in offsets]
	return offsets, max(offsets)

def resolve(measurement, pixelOffsets):  #pId -> offset
	conversion.clearDict()
	global bestLoss
	global bestParams
	bestLoss = sys.float_info.max
	bestParams = []

	pId = [] #Parallel arrays
	offsets = [] #Offset can have negative numbers
	for i in pixelOffsets:
		pId.append(i)
		offsets.append(pixelOffsets[i])

	offsets, maxValue = normalizeOffsets(offsets)

	def lossFunction(params):
		residue = measurement.copy()
		for i in range(len(offsets)):
			residue = residue - conversion.createCurvePixelSpace(params[i], offsets[i], maxValue)  # Can't use -=
		s = np.sum(np.abs(residue))

		#plt.clf()
		#plt.plot(residue)
		#plt.show()
		return s

	#t = time.time_ns()
	solution = solveLeastSquares(offsets, lossFunction)
	#print((time.time_ns() - t) / 1000000000)
	if len(offsets) < 5:
		#t = time.time_ns()
		bruteForceSolution = solveBruteForce(offsets, lossFunction)
		if lossFunction(bruteForceSolution) < lossFunction(solution):
			solution = bruteForceSolution

	#print((time.time_ns() - t) / 1000000000)

	resultDict = dict()
	ideal = sum([conversion.createCurvePixelSpace(solution[i], offsets[i], maxValue) for i in range(len(offsets))])
	for i in range(len(offsets)): #Use proportion of theoretical to weight measurement
		demixed = (conversion.createCurvePixelSpace(solution[i], offsets[i], maxValue) / ideal) * measurement
		resultDict[pId[i]] = int(solution[i]), conversion.convertPixelToPhysical(demixed[offsets[i]:offsets[i] + calibration.PIXEL_END])
	return resultDict

def solveLeastSquares(offsets, lossFunction): #xOffsets
	initialGuess = []
	lowerBound = []
	upperBound = []
	for _ in offsets:
		initialGuess.append((calibration.MIN_TEMP + calibration.MAX_TEMP) / 2)
		lowerBound.append(calibration.MIN_TEMP)
		upperBound.append(calibration.MAX_TEMP)

	return least_squares(lossFunction, x0=initialGuess, bounds=(lowerBound, upperBound)).x


def solveBruteForce(offsets, lossFunction):
	minTemps = [calibration.MIN_TEMP for i in offsets]
	maxTemps = [calibration.MAX_TEMP for i in offsets]

	for iterations in range(3):
		combinationTempHelper([], minTemps, maxTemps, lossFunction, len(offsets))
		minTemps, maxTemps = getNewBounds(minTemps, maxTemps)

	return bestParams

def combinationTempHelper(paramList, minTemp, maxTemp, lossFunction, numOffsets):
	k = len(paramList)
	for i in range(tempGranularity + 1):
		paramList.append(minTemp[k] + (maxTemp[k] - minTemp[k]) * i / tempGranularity)
		if len(paramList) == numOffsets:
			v = lossFunction(paramList)
			if v == -1:
				del paramList[-1]
				return
			global bestLoss
			global bestParams
			if v < bestLoss:
				bestLoss = v
				bestParams = paramList.copy()
			del paramList[-1]
		else:
			combinationTempHelper(paramList, minTemp, maxTemp, lossFunction, numOffsets)
			del paramList[-1]

def getNewBounds(minTemps, maxTemps):
	tempDiff = (maxTemps[0] - minTemps[0]) / tempGranularity

	minTemps = [max(minTemps[i], bestParams[i] - 2 * tempDiff) for i in range(len(bestParams))]
	maxTemps = [min(maxTemps[i], bestParams[i] + 2 * tempDiff) for i in range(len(bestParams))]
	return minTemps, maxTemps

'''
testIds = [0, 1, 2, 3]
testTemps = [2350, 2750, 2350, 2520]
testOffsets = [-10, 30, 0, 90]

#testIds = [3]
#testTemps = [2720]
#testOffsets = [0]

normalizedTestOffsets, maxTestOffset = normalizeOffsets(testOffsets)
curves = [conversion.createCurvePixelSpace(testTemps[i], normalizedTestOffsets[i], maxTestOffset) for i in range(len(testTemps))]
offsetDict = dict()
for i in range(len(testIds)):
	offsetDict[testIds[i]] = normalizedTestOffsets[i]

result = resolve(sum(curves), offsetDict)
for pId in result:
	print(str(pId) + ": " + str(result[pId][0]))
'''
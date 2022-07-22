import numpy as np
import utils
import sys


measurement = []
offsets = []

tempGranularity = 15

bestLoss = sys.float_info.max
bestParams = []

def bruteLossFunction(temps):
	residue = measurement.copy()
	for i in range(len(offsets)):
		residue = residue - utils.createCurve(temps[i], offsets[i], offsets[-1]) #Can't use -=
	#if np.amax(residue) < 0:
	#	return -1
	return np.sum(np.absolute(residue))


def combinationTempHelper(paramList, minTemp, maxTemp):
	tooLargeFlag = False
	k = len(paramList)
	for i in range(tempGranularity + 1):
		paramList.append(minTemp[k] + (maxTemp[k] - minTemp[k]) * i / tempGranularity)
		if len(paramList) == len(offsets):
			v = bruteLossFunction(paramList)
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
			combinationTempHelper(paramList, minTemp, maxTemp)
			del paramList[-1]

def getNewBounds(minTemps, maxTemps):
	tempDiff = (maxTemps[0] - minTemps[0]) / tempGranularity

	minTemps = [max(minTemps[i], bestParams[i] - 2 * tempDiff) for i in range(len(bestParams))]
	maxTemps = [min(maxTemps[i], bestParams[i] + 2 * tempDiff) for i in range(len(bestParams))]
	return minTemps, maxTemps

def optimize(pixelOffsets, combinedIntensities):
	global measurement
	global offsets
	global whereMask
	measurement = combinedIntensities
	offsets = pixelOffsets

	minTemps = [utils.minTemp for i in offsets]
	maxTemps = [utils.maxTemp for i in offsets]

	for iterations in range(3):
		#print(minTemps)
		#print(maxTemps)
		combinationTempHelper([], minTemps, maxTemps)
		#print(bestParams)
		#print(bestLoss)
		#print()
		minTemps, maxTemps = getNewBounds(minTemps, maxTemps)

	return bestParams

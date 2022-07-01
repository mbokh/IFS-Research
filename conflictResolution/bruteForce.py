import numpy as np
import utils
import sys


measurement = []
offsets = []

tempGranularity = 10
gainGranularity = 5

bestLoss = sys.float_info.max
bestParams = []

def bruteLossFunction(temps):
	residue = measurement.copy()
	for i in range(len(offsets)):
		residue = residue - utils.createCurve(temps[2 * i], temps[(2 * i) + 1], offsets[i], offsets[-1]) #Can't use -=
	#if np.amax(residue) < 0:
	#	return -1
	return np.sum(np.absolute(residue))


def combinationGainHelper(paramList, minTemp, maxTemp, minGain, maxGain):
	tooLargeFlag = False
	k = int(len(paramList) / 2)
	for i in range(gainGranularity + 1):
		paramList.append(minGain[k] + (maxGain[k] - minGain[k]) * i / gainGranularity)
		if len(paramList) == len(offsets) * 2:
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
			combinationTempHelper(paramList, minTemp, maxTemp, minGain, maxGain)
			del paramList[-1]


def combinationTempHelper(paramList, minTemp, maxTemp, minGain, maxGain):
	k = int(len(paramList) / 2)
	tooLargeFlag = False
	for i in range(tempGranularity + 1):
		paramList.append(minTemp[k] + (maxTemp[k] - minTemp[k]) * i / tempGranularity)
		combinationGainHelper(paramList, minTemp, maxTemp, minGain, maxGain)
		del paramList[-1]




def getNewBounds(minTemps, maxTemps, minGain, maxGain):
	tempDiff = (maxTemps[0] - minTemps[0]) / tempGranularity
	gainDiff = (maxGain[0] - minGain[0]) / gainGranularity

	minTemps = [max(minTemps[i // 2], bestParams[i] - 2 * tempDiff) for i in range(0, len(bestParams), 2)]
	maxTemps = [min(maxTemps[i // 2], bestParams[i] + 2 * tempDiff) for i in range(0, len(bestParams), 2)]
	minGain = [max(minGain[i // 2], bestParams[i + 1] - 2 * gainDiff) for i in range(0, len(bestParams), 2)]
	maxGain = [min(maxGain[i // 2], bestParams[i + 1] + 2 * gainDiff) for i in range(0, len(bestParams), 2)]
	return minTemps, maxTemps, minGain, maxGain

def optimize(pixelOffsets, combinedIntensities):
	global measurement
	global offsets
	global whereMask
	measurement = combinedIntensities
	offsets = pixelOffsets

	minTemps = [utils.minTemp for i in offsets]
	maxTemps = [utils.maxTemp for i in offsets]
	minGain = [utils.minGain for i in offsets]
	maxGain = [utils.maxGain for i in offsets]

	for iterations in range(3):
		print(minTemps)
		print(maxTemps)
		print(minGain)
		print(maxGain)
		combinationTempHelper([], minTemps, maxTemps, minGain, maxGain)
		print(bestParams)
		print(bestLoss)
		print()
		minTemps, maxTemps, minGain, maxGain = getNewBounds(minTemps, maxTemps, minGain, maxGain)

	return bestParams

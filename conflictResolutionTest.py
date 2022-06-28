import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import sys

lambdaStart = 0.0000004
lambdaEnd = 0.0000008
pixelEnd = 300 #Exclusive
pixelX = np.linspace(0, pixelEnd - 1, pixelEnd)

c1 = 1.19 / (10**16)
c2 = 0.0144

tempGranularity = 20
gainGranularity = 30

bestLoss = sys.float_info.max
bestParams = []

def pixelToWavelength(pixels):
	return lambdaStart + (lambdaEnd - lambdaStart)*(pixels / pixelEnd)

def plancksLaw(wavelengths, T):
	return (c1 / np.power(wavelengths, 5)) / (np.exp(c2 / (wavelengths * T)) - 1)

def identityResponse(i):
	return i

def gaussianResponse(i):
	return np.exp(-0.5 * np.square( (i - (pixelEnd/2)) / (pixelEnd / 5) ))

def createCurve(T, gain, pixelShift):
	i = plancksLaw(pixelToWavelength(pixelX), T) * gain

	shiftedI = np.zeros(pixelEnd + groundTruth[-1][2])
	shiftedI[pixelShift:pixelEnd + pixelShift] = i * responseFunction(pixelX)
	return shiftedI

def lossFunction(temps):
	residue = combinedIntensities
	for i in range(len(groundTruth)):
		residue = residue - createCurve(temps[2 * i], temps[(2 * i) + 1], groundTruth[i][2]) #Can't use -=
	a = np.sum(np.absolute(residue))
	return a

def bruteLossFunction(temps):
	residue = combinedIntensities
	for i in range(len(groundTruth)):
		residue = residue - createCurve(temps[2 * i], temps[(2 * i) + 1], groundTruth[i][2]) #Can't use -=
	a = np.sum(np.square(residue))
	return a

def combinationGainHelper(paramList, minTemp, maxTemp, minGain, maxGain):
	k = int(len(paramList) / 2)
	for i in range(gainGranularity + 1):
		paramList.append(minGain[k] + (maxGain[k] - minGain[k]) * i / gainGranularity)
		if len(paramList) == len(groundTruth) * 2:
			v = bruteLossFunction(paramList)
			global bestLoss
			global bestParams
			if v < bestLoss:
				bestLoss = v
				bestParams = paramList.copy()
		else:
			combinationTempHelper(paramList, minTemp, maxTemp, minGain, maxGain)
		del paramList[-1]

def combinationTempHelper(paramList, minTemp, maxTemp, minGain, maxGain):
	k = int(len(paramList) / 2)
	for i in range(tempGranularity + 1):
		paramList.append(minTemp[k] + (maxTemp[k] - minTemp[k]) * i / tempGranularity)
		combinationGainHelper(paramList, minTemp, maxTemp, minGain, maxGain)
		del paramList[-1]


def findBestCombination():
	combinationTempHelper([], [2000 for i in groundTruth], [3000 for i in groundTruth], [0 for i in groundTruth], [1 for i in groundTruth])
	minTemps = [bestParams[i] - 2 * (1000 / tempGranularity) for i in range(0, len(bestParams), 2)]
	maxTemps = [bestParams[i] + 2 * (1000 / tempGranularity) for i in range(0, len(bestParams), 2)]
	minGain = [bestParams[i + 1] - 2 * (1.0 / gainGranularity) for i in range(0, len(bestParams), 2)]
	maxGain = [bestParams[i + 1] + 2 * (1.0 / gainGranularity) for i in range(0, len(bestParams), 2)]
	print(bestParams)
	print(bestLoss)
	combinationTempHelper([], minTemps, maxTemps, minGain, maxGain)
	print(bestParams)
	print(bestLoss)

groundTruth = [(2123, 0.3, 0), (2650, 0.73, 140)]

responseFunction = gaussianResponse

curves = [createCurve(data[0], data[1], data[2]) for data in groundTruth]
combinedIntensities = sum(curves)

combinedX = np.linspace(0, pixelEnd + groundTruth[-1][2] - 1, pixelEnd + groundTruth[-1][2])
fig, axs = plt.subplots(2, 1)
for c in curves:
	axs[0].plot(combinedX, c)

axs[1].plot(combinedX, combinedIntensities, 'g')

fig.tight_layout()
plt.show()


#[2700.0, 0.05, 2700.0, 0.65]
#print(bruteLossFunction([2440, 1, 2650, 1]))
findBestCombination()
#print(bestParams)
#print(bestLoss)
'''
initialGuess = []
lowerBound = []
upperBound = []
for i in groundTruth:
	initialGuess.extend([2500, 1])
	lowerBound.extend([2000, 0])
	upperBound.extend([3000, 4])

solution = least_squares(lossFunction, x0=initialGuess, bounds=(lowerBound, upperBound)).x
print(solution)'''
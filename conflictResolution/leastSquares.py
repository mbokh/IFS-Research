import numpy as np
import utils
from scipy.optimize import least_squares

measurement = []
offsets = []

def lossFunction(params):
	residue = measurement.copy()
	for i in range(len(offsets)):
		residue = residue - utils.createCurve(params[2 * i], params[(2 * i) + 1], offsets[i], offsets[-1]) #Can't use -=
	return np.sum(np.absolute(residue))

def optimize(pixelOffsets, combinedIntensities):
	global measurement
	global offsets
	measurement = combinedIntensities
	offsets = pixelOffsets

	initialGuess = []
	lowerBound = []
	upperBound = []
	for _ in offsets:
		initialGuess.extend([(utils.minTemp + utils.maxTemp) / 2, (utils.minGain + utils.maxGain) / 2])
		lowerBound.extend([utils.minTemp, utils.minGain])
		upperBound.extend([utils.maxTemp, utils.maxGain])

	return least_squares(lossFunction, x0=initialGuess, bounds=(lowerBound, upperBound)).x
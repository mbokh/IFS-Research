import numpy as np
import utils
from scipy.optimize import least_squares


def optimize(offsets, measurement):
	initialGuess = []
	lowerBound = []
	upperBound = []
	for _ in offsets:
		initialGuess.append((utils.minTemp + utils.maxTemp) / 2)
		lowerBound.append(utils.minTemp)
		upperBound.append(utils.maxTemp)

	def lossFunction(params):
		residue = measurement.copy()
		for i in range(len(offsets)):
			residue = residue - utils.createCurve(params[i], offsets[i], offsets[-1])  # Can't use -=
		return np.sum(np.absolute(residue))

	return least_squares(lossFunction, x0=initialGuess, bounds=(lowerBound, upperBound)).x
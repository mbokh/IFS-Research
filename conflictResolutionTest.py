import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

lambdaStart = 0.0000004
lambdaEnd = 0.0000008
pixelEnd = 300 #Exclusive
pixelX = np.linspace(0, pixelEnd - 1, pixelEnd)

c1 = 1.19 / (10**16)
c2 = 0.0144

def pixelToWavelength(pixels):
	return lambdaStart + (lambdaEnd - lambdaStart)*(pixels / pixelEnd)

def plancksLaw(wavelengths, T):
	return (c1 / np.power(wavelengths, 5)) / (np.exp(c2 / (wavelengths * T)) - 1)

def identityResponse(i):
	return i

def gaussianResponse(i):
	return np.exp(-0.5 * np.square( (i - (pixelEnd/2)) / (pixelEnd / 5) ))

def createCurve(T, pixelShift):
	i = plancksLaw(pixelToWavelength(pixelX), T)

	shiftedI = np.zeros(pixelEnd + offset[-1])
	shiftedI[pixelShift:pixelEnd + pixelShift] = i * responseFunction(pixelX)
	return shiftedI

def lossFunction(temps):
	residue = combinedIntensities
	for i in range(len(groundTruth)):
		residue = residue - createCurve(temps[i], offset[i]) #Can't use -=
	a = np.sum(np.absolute(residue))
	return a

groundTruth = [2640, 2660, 2650]
offset = [0, 20, 25]
responseFunction = gaussianResponse

curves = [createCurve(groundTruth[i], offset[i]) for i in range(len(groundTruth))]
combinedIntensities = sum(curves)

combinedX = np.linspace(0, pixelEnd + offset[-1] - 1, pixelEnd + offset[-1])
fig, axs = plt.subplots(2, 1)
for c in curves:
	axs[0].plot(combinedX, c)

axs[1].plot(combinedX, combinedIntensities, 'g')

fig.tight_layout()
plt.show()

solution = least_squares(lossFunction, x0=[2500 for i in range(len(groundTruth))],
					bounds=([2000 for i in range(len(groundTruth))],
							[3000 for i in range(len(groundTruth))]),
					diff_step=0.0001)
print(solution)

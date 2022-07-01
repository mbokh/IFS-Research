import numpy as np

lambdaStart = 0.0000004
lambdaEnd = 0.0000008
pixelEnd = 300 #Exclusive
pixelX = np.linspace(0, pixelEnd - 1, pixelEnd)

minTemp = 2000.0
maxTemp = 3000.0
minGain = 0.0
maxGain = 1.0

c1 = 1.19 / (10**16)
c2 = 0.0144

tempLookup = dict()

def pixelToWavelength(pixels):
	return lambdaStart + (lambdaEnd - lambdaStart)*(pixels / pixelEnd)

def plancksLaw(wavelengths, T):
	return (c1 / np.power(wavelengths, 5)) / (np.exp(c2 / (wavelengths * T)) - 1)

def createCurve(T, gain, pixelShift, maxOffset):
	if (T, pixelShift) in tempLookup.keys():
		return tempLookup[(T, pixelShift)] * gain
	i = plancksLaw(pixelToWavelength(pixelX), T)

	shiftedI = np.zeros(pixelEnd + maxOffset)
	shiftedI[pixelShift:pixelEnd + pixelShift] = i * responseFunction(pixelX)
	tempLookup[(T, pixelShift)] = shiftedI
	return shiftedI * gain

def identityResponse(i):
	return i

def gaussianResponse(i):
	return np.exp(-0.5 * np.square( (i - (pixelEnd/2)) / (pixelEnd / 5) ))

responseFunction = gaussianResponse

import numpy as np
from backend import calib as calibration


def convertPhysicalToPixel(planckData):
	return planckData * calibration.SYSTEM_RESPONSE

def convertPixelToPhysical(pixelData):
	return pixelData / calibration.SYSTEM_RESPONSE


c1 = 1.19 / (10**16)
c2 = 0.0144

tempLookup = dict()

def clearDict(): #When starting a new cluster, offsets are different, so have to clear cache, faster than keying on temp only
	tempLookup.clear()

def plancksLaw(wavelengths, T):
	return (c1 / np.power(wavelengths, 5)) / (np.exp(c2 / (wavelengths * T)) - 1)

#Returns the appropriate response-shaped curve in pixel space in both x and y
def createCurvePixelSpace(T, pixelShift, maxOffset):
	if (T, pixelShift) in tempLookup.keys(): #Lookup needs to be float, not int, or else small steps of least-squares will round to same int
		return tempLookup[(T, pixelShift)]
	raw = plancksLaw(calibration.PIXEL_TO_WAVELENGTH, T)

	shiftedI = np.zeros(calibration.PIXEL_END + maxOffset)
	shiftedI[pixelShift:calibration.PIXEL_END + pixelShift] = convertPhysicalToPixel(raw)
	tempLookup[(T, pixelShift)] = shiftedI
	return shiftedI

'''
temps = [2350, 2750, 2350, 2520]
offsets = [0, 10, 40, 50]

curves = [createCurvePixelSpace(temps[i], offsets[i], offsets[-1]) for i in range(len(temps))]
combinedIntensities = sum(curves)

combinedX = np.linspace(0, calibration.PIXEL_END + offsets[-1] - 1, calibration.PIXEL_END + offsets[-1])
fig, axs = plt.subplots(2, 1)
for c in curves:
	axs[0].plot(combinedX, c)
axs[1].plot(combinedX, combinedIntensities, 'g')
fig.tight_layout()
plt.show()'''

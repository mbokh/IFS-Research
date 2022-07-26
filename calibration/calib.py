import numpy as np

def getExtendsLeft():
	return 270

def getExtendsRight():
	return 150

def pixelToWavelength():
	return lambdaStart + (lambdaEnd - lambdaStart)*(pixelX / pixelEnd)

def cameraResponse():
	return camResponse


lambdaStart = 0.0000004
lambdaEnd = 0.0000008
pixelEnd = getExtendsRight() + getExtendsLeft() #Exclusive
pixelX = np.linspace(0, pixelEnd - 1, pixelEnd)
minTemp = 2000.0
maxTemp = 3000.0
camResponse = np.exp(-0.5 * np.square( (pixelX - (pixelEnd/2)) / (pixelEnd / 5) )) * 10**(-10)


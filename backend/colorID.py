import math
import random

colors = dict()

def getNewColor():
	while True:
		c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		if c[2] > 120 and c[0] < 80 and c[1] < 80: #Too close to pure red
			continue
		if math.sqrt(c[0]**2 + c[1]**2 + c[2]**2) > 200:
			return c

def getColorOfId(boxId):
	if not boxId in colors.keys():
		colors[boxId] = getNewColor()
	return colors[boxId]


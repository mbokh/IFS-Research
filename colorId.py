import math
import random

class ColorId:
	def __init__(self):
		self.colors = dict()

	def getColorOfId(self, boxId):
		if not boxId in self.colors.keys():
			self.colors[boxId] = getNewColor()
		return self.colors[boxId]


def getNewColor():
	while True:
		c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		if math.sqrt(c[0]**2 + c[1]**2 + c[2]**2) > 200:
			return c

def addToDict(d, key, value):
	if not key in d.keys():
		d[key] = [value]
	else:
		d[key].append(value)

class Logger:
	def __init__(self):
		self.byFrame = dict()
		self.byParticleId = dict()
		self.data = []

	def logData(self, track, frameNum):
		boundingData = track.getTrackingData()
		for particleId, (x, y, w, h, cX, cY), occCount in boundingData:
			self.data.append((int(cX), int(cY), frameNum, particleId, occCount))
			addToDict(self.byFrame, frameNum, len(self.data) - 1)
			addToDict(self.byParticleId, particleId, len(self.data) - 1)
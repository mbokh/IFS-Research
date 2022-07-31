import cv2

# -------------------------Width
# |                  |
# |                  |
# |     spectra      |  real
# |      lines       |  image
# |                  |
# sStart            sEnd
class VideoSource:
	def __init__(self, filename, skip, end, spectraStart, spectraEnd):
		self.video = cv2.VideoCapture(filename)
		for i in range(skip):
			ret, frame = self.video.read()
		self.frameNum = 0
		self.end = end
		self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.spectraStart = spectraStart
		self.spectraEnd = spectraEnd

	def getFrame(self):
		self.frameNum += 1
		ret, f = self.video.read()
		if not ret or self.frameNum == self.end:
			return None, self.frameNum
		return f, self.frameNum

	def getWidth(self):
		return self.width

	def getHeight(self):
		return self.height

	def getSpectraPartition(self):
		return self.spectraStart, self.spectraEnd

	def destroy(self):
		self.video.release()
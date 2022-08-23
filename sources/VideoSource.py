import cv2
import numpy as np
# -------------------------Width
# |                  |
# |                  |
# |     spectra      |  real
# |      lines       |  image
# |                  |
# sStart            sEnd
class VideoSource:
	def __init__(self, filename, skip, end, spectraStart, spectraEnd, flipLR):
		self.video = cv2.VideoCapture("Sources/" + filename)
		for i in range(skip):
			ret, frame = self.video.read()
		self.frameNum = 0
		self.end = end
		self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.spectraStart = max(0, min(spectraStart, self.width - 1))
		self.spectraEnd = max(0, min(spectraEnd, self.width - 1))
		self.flipLR = flipLR

	def getFrame(self):
		self.frameNum += 1
		ret, f = self.video.read()
		if not ret or self.frameNum == self.end:
			return None, self.frameNum
		return (cv2.flip(f, 1) if self.flipLR else f), self.frameNum

	def getWidth(self):
		return self.width

	def getHeight(self):
		return self.height

	def getSpectraPartition(self):
		return ((self.getWidth() - 1) - self.spectraEnd, (self.getWidth() - 1) - self.spectraStart) if self.flipLR else (self.spectraStart, self.spectraEnd)

	def destroy(self):
		self.video.release()
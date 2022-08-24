import cv2

# -------------------------Width
# |                  |
# |                  |
# |     spectra      |  real
# |      lines       |  image
# |                  |
# sStart            sEnd
def getStringedNumber(x):
	if x < 10:
		return "00000" + str(x)
	if x < 100:
		return "0000" + str(x)
	if x < 1000:
		return "000" + str(x)
	if x < 10000:
		return "00" + str(x)
	if x < 100000:
		return "0" + str(x)
	return str(x)

def getFilePath(fileNum, prefix):
	return "Sources/" + prefix + "/" + prefix + getStringedNumber(fileNum) + ".tif"

class FramesSource:
	def __init__(self, prefix, skip, end, spectraStart, spectraEnd, flipLR):
		self.prefix = prefix
		self.fileNum = skip + 1
		self.frameNum = 0
		self.end = end
		frame = cv2.imread(getFilePath(1, prefix))
		self.height = int(frame.shape[0])
		self.width = int(frame.shape[1])
		self.spectraStart = spectraStart
		self.spectraEnd = spectraEnd
		self.flipLR = flipLR

	def getFrame(self):
		self.frameNum += 1
		f = cv2.imread(getFilePath(self.fileNum, self.prefix))
		self.fileNum += 1
		if f is None or self.frameNum == self.end:
			return None, self.frameNum
		return (cv2.flip(f, 1) if self.flipLR else f), self.frameNum

	def getWidth(self):
		return self.width

	def getHeight(self):
		return self.height

	def getSpectraPartition(self):
		return ((self.getWidth() - 1) - self.spectraEnd, (self.getWidth() - 1) - self.spectraStart) if self.flipLR else (self.spectraStart, self.spectraEnd)

	def destroy(self):
		return #Dummy method just to have method signatures identical to VideoSource
from sources import VideoSource
from backend import calib, database, spectraExtract, tracking

import pickle
import time


startTime = time.time_ns()

video = VideoSource.VideoSource(filename="Al3Zr_SM_30k_Run2.avi", skip=0, end=-1, spectraStart=150, spectraEnd=1023, flipLR=True)
#video = FramesSource.FramesSource(prefix="Al3Zr_SM_30k_30k_sh_g15_5mm_toff_0ms_Run5", skip=0, end=-1, spectraStart=150, spectraEnd=1023, flipLR=True)
tracker = tracking.MultiObjectTracker()

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break
	if frameNum % 50 == 0:
		print(frameNum)

	tracker.processImage(frame, frameNum, video)
	f = spectraExtract.extractRawSpectra(frame, video)

video.destroy()

print("Extraction took "  + str((time.time_ns() - startTime) / 1000000000) + " seconds")

data = (calib.MIN_WAVELENGTH, calib.MAX_WAVELENGTH, database.getFullDataForPickling(video.getWidth()))
with open('extractedData/extractedDataRestrictedRange.pickle', 'wb') as pickleFile:
	pickle.dump(data, pickleFile)
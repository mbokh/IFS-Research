import database
import tracking
from Sources import VideoSource
from Sources import FramesSource
import spectraExtract
import calib

import pickle
import time


startTime = time.time_ns()

video = VideoSource.VideoSource(filename="Sources/Al3Zr_SM_30k_Run2.avi", skip=0, end=-1, spectraStart=150, spectraEnd=1024 - 1)
#video = FramesSource.FramesSource(prefix="Videos/Al3Zr_SM_30k_30k_sh_g15_5mm_toff_0ms_Run2/Al3Zr_SM_30k_30k_sh_g15_5mm_toff_0ms_Run2", skip=0, end=-1, spectraStart=150, spectraEnd=1024 - 1)
tracker = tracking.MultiObjectTracker()

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break
	if frameNum % 50 == 0:
		print(frameNum)

	tracker.processImage(frame, frameNum, video)
	f, spectra = spectraExtract.extractRawSpectra(frame, video)

video.destroy()

print("Extraction took "  + str((time.time_ns() - startTime) / 1000000000) + " seconds")

data = (calib.MIN_WAVELENGTH, calib.MAX_WAVELENGTH, database.getFullDataForPickling())
with open('extractedDataMedianBlurThresh9.pickle', 'wb') as pickleFile:
	pickle.dump(data, pickleFile)
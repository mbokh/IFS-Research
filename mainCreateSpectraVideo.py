import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import numpy as np
import pickle

from backend import calib as calibration, conversion

DUMMY_PARTICLE = -2

def getSpectraDataForFrame(frameNum, particles):
	lastFrame = True
	data = dict()
	for pId in particles:
		p = particles[pId]

		if p.frameNumAppeared > frameNum: #Current frame is too early
			lastFrame = False
			continue
		if p.frameNumAppeared + len(p.particleData) <= frameNum: #Current frame past end
			continue

		data[pId] = p.spectraData[frameNum - p.frameNumAppeared]

	if lastFrame and len(data) > 0:
		lastFrame = False
	return data, lastFrame


def formatter(x, pos):
	return '%0.f' % (x*1e9)


def setIDOrderList(futureIds, previousList):
	for i in range(len(previousList)):
		if previousList[i] not in futureIds:
			previousList[i] = DUMMY_PARTICLE

	for pId in sorted(list(futureIds)):
		if pId not in previousList:
			if DUMMY_PARTICLE in previousList:
				previousList[previousList.index(DUMMY_PARTICLE)] = pId
			else:
				previousList.append(pId)



matplotlib.use('TkAgg')
#graph = plt.figure()
ROWS = 4
COLS = 5
fig, axs = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(16.25, 11.5))
fig.canvas.manager.window.wm_geometry("+%d+%d" % (2000, 0))

outVideo = None

plt.ion()
plt.show()

with open('extractedData/extractedDataRestrictedRange.pickle', 'rb') as f:
	minWavelength, maxWavelength, particleData = pickle.load(f)
	frameNum = 0

	idOrderedList = []
	while True:
		#fig.clf()
		data, lastFrame = getSpectraDataForFrame(frameNum, particleData)
		if lastFrame:
			break

		graphNum = 0

		setIDOrderList(data.keys(), idOrderedList)

		for pId in idOrderedList:
			if graphNum > ROWS * COLS - 1 or pId == DUMMY_PARTICLE:
				break
			temp, spectra, code = data[pId]

			color = "#1dd13b" #Green
			if code == 1:
				color = "#1a29f0" #Blue
			if code == 2:
				color = "#9bf707" #Light green
			if code == 3:
				color = "#07eff7" #Light blue
			if code == 4:
				color = "#f70707"
			axs[int(graphNum / COLS), graphNum % COLS].clear()
			xs = np.linspace(minWavelength, maxWavelength, len(spectra))
			axs[int(graphNum / COLS), graphNum % COLS].plot(xs, spectra, label=("" + str(pId)), color=color)

			theoreticalCurve = conversion.plancksLaw(calibration.PIXEL_TO_WAVELENGTH, temp)
			axs[int(graphNum / COLS), graphNum % COLS].plot(xs, theoreticalCurve, color='black', linestyle='--', linewidth=0.7)

			axs[int(graphNum / COLS), graphNum % COLS].set_title("ID: " + str(pId) + ", Temp: " + str(temp))
			axs[int(graphNum / COLS), graphNum % COLS].set_xlabel('Wavelength (nm)')
			axs[int(graphNum / COLS), graphNum % COLS].set_ylim(0, 1 * (10**12))
			axs[int(graphNum / COLS), graphNum % COLS].set_xlim(5.5 * 1e-7, 8 * 1e-7) #Dummy bounds
			axs[int(graphNum / COLS), graphNum % COLS].xaxis.set_major_formatter(ticks.FuncFormatter(formatter))

			graphNum += 1
		while graphNum < ROWS * COLS:
			axs[int(graphNum / COLS), graphNum % COLS].clear()
			axs[int(graphNum / COLS), graphNum % COLS].set_title("")
			axs[int(graphNum / COLS), graphNum % COLS].set_xlabel('Wavelength (nm)')
			axs[int(graphNum / COLS), graphNum % COLS].set_ylim(0, 1 * (10 ** 12))
			axs[int(graphNum / COLS), graphNum % COLS].set_xlim(5.5 * 1e-7, 8 * 1e-7)  # Dummy bounds
			axs[int(graphNum / COLS), graphNum % COLS].xaxis.set_major_formatter(ticks.FuncFormatter(formatter))
			graphNum += 1

		fig.suptitle("Frame Number: " + str(frameNum))
		plt.draw()
		fig.tight_layout()
		fig.canvas.flush_events()


		if outVideo is None:
			s = fig.get_size_inches()*fig.dpi
			outVideo = cv2.VideoWriter("extractedData/spectraVideoRestricted.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 10, (int(s[0]), int(s[1])))
		img = np.array(fig.canvas.buffer_rgba())
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
		outVideo.write(img)

		plt.pause(0.01)
		frameNum += 1

cv2.destroyAllWindows()
outVideo.release()
plt.close('all')
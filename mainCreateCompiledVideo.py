import cv2
import matplotlib
import matplotlib.pyplot as plt
import colorID

import Sources.VideoSource as VideoSource
import pickle
import numpy as np
import math


def decorateFrame(particles, f, frameNum, showDebugInfo=True, showPath=True, fullBoundingBox=False):
	numParticles = 0
	lastParticleId = 0
	for pId in particles:
		p = particles[pId]

		if p.frameNumAppeared > frameNum: #Current frame is too early
			continue
		if p.frameNumAppeared + len(p.particleData) <= frameNum: #Current frame past end
			continue

		numParticles += 1
		lastParticleId = max(lastParticleId, pId)

		bBox, brightness, occCount = p.particleData[frameNum - p.frameNumAppeared]
		x, y, w, h, cX, cY = coordTransform(bBox)

		# Particle coloring
		colorRegionWithTemperature(f, p.spectraData[frameNum - p.frameNumAppeared][0], (x, y, w, h, cX, cY))

		if occCount > 0:
			cv2.putText(f, str(occCount), (x - 15, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA, False)
		if fullBoundingBox:
			cv2.rectangle(f, (x, y), (x + w, y + h), colorID.getColorOfId(pId) if occCount == 0 else (0, 0, 255), 1)
		else:
			cv2.circle(f, (x, y), 1, colorID.getColorOfId(pId) if occCount == 0 else (0, 0, 255), -1)
			cv2.circle(f, (x, y + h), 1, colorID.getColorOfId(pId) if occCount == 0 else (0, 0, 255), -1)
			cv2.circle(f, (x + w, y), 1, colorID.getColorOfId(pId) if occCount == 0 else (0, 0, 255), -1)
			cv2.circle(f, (x + w, y + h), 1, colorID.getColorOfId(pId) if occCount == 0 else (0, 0, 255), -1)

		cv2.putText(f, str(pId), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if occCount == 0 else (0, 0, 255), 1, cv2.LINE_AA, False)

		if showPath:
			for i in range(frameNum - p.frameNumAppeared + 1):
				bBox, brightness, occCount = p.particleData[i]
				x, y, w, h, cX, cY = coordTransform(bBox)

				cv2.circle(f, (x, y), 2, (0, 0, 255) if occCount > 0 else colorID.getColorOfId(pId), -1)
				if i < frameNum - p.frameNumAppeared:
					nextPointCoords = coordTransform(p.particleData[i + 1][0])
					cv2.line(f, (x, y), (nextPointCoords[0], nextPointCoords[1]), colorID.getColorOfId(pId), 1)


	if showDebugInfo:
		cv2.putText(f, "Frame Num: " + str(frameNum), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
		cv2.putText(f, "Last ID used: " + str(lastParticleId), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
		cv2.putText(f, "Particles: " + str(numParticles), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

	return f

def hexString(color):
	return '#%02x%02x%02x' % (color[2], color[1], color[0]) #Backwards for opencv's bgr

def makeTempGraph(particles, frameNum):
	plt.clf()
	plt.gca().set_xlim(0, 100)
	plt.gca().set_xlabel('Frames Elapsed')
	plt.gca().set_ylim(2000, 3000)
	for pId in particles:
		p = particles[pId]

		if p.frameNumAppeared > frameNum: #Current frame is too early
			continue
		if p.frameNumAppeared + len(p.particleData) <= frameNum: #Current frame past end
			continue

		temperature = list(list(zip(*p.spectraData[:frameNum - p.frameNumAppeared + 1]))[0])
		temperature.reverse()

		plt.plot(temperature, color=hexString(colorID.getColorOfId(pId)), linewidth=0.7)

	plt.title("Temperature")
	plt.draw()
	graph.canvas.flush_events()
	plt.pause(0.01)


def makeSpectraGraph(particles, frameNum, minWavelength, maxWavelength):
	plt.clf()
	plt.gca().set_xlabel('Wavelength')
	plt.gca().set_ylim(0, 5 * (10**12))
	for pId in particles:
		p = particles[pId]

		if p.frameNumAppeared > frameNum: #Current frame is too early
			continue
		if p.frameNumAppeared + len(p.particleData) <= frameNum: #Current frame past end
			continue

		spectra = p.spectraData[frameNum - p.frameNumAppeared][1]
		xs = np.linspace(minWavelength, maxWavelength, len(spectra))

		plt.plot(xs, spectra, color=hexString(colorID.getColorOfId(pId)), linewidth=0.7)

	plt.title("Spectra")
	plt.draw()
	graph.canvas.flush_events()
	plt.pause(0.01)


# THIS FUNCTION IS TAKEN FROM https://gist.github.com/petrklus/b1f427accdf7438606a6 AND MODIFIED FOR MY USE
def convert_K_to_RGB(colour_temperature):
	"""
	Converts from K to RGB, algorithm courtesy of
	http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
	"""
	# range check
	if colour_temperature < 1000:
		colour_temperature = 1000
	elif colour_temperature > 40000:
		colour_temperature = 40000

	tmp_internal = colour_temperature / 100.0

	# red
	if tmp_internal <= 66:
		red = 255
	else:
		tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
		if tmp_red < 0:
			red = 0
		elif tmp_red > 255:
			red = 255
		else:
			red = tmp_red

	# green
	if tmp_internal <= 66:
		tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
		if tmp_green < 0:
			green = 0
		elif tmp_green > 255:
			green = 255
		else:
			green = tmp_green
	else:
		tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
		if tmp_green < 0:
			green = 0
		elif tmp_green > 255:
			green = 255
		else:
			green = tmp_green

	# blue
	if tmp_internal >= 66:
		blue = 255
	elif tmp_internal <= 19:
		blue = 0
	else:
		tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
		if tmp_blue < 0:
			blue = 0
		elif tmp_blue > 255:
			blue = 255
		else:
			blue = tmp_blue

	return int(red), int(green), int(blue)

def getConvertedColorHue(T):
	#To increase viewing contrast, linearly maps 2000 -> 1500, 3000 -> 3500
	r, g, b = convert_K_to_RGB((2 * (T - 3000)) + 3500)
	return cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)

def colorRegionWithTemperature(img, T, bBox):
	region = img[bBox[1]:bBox[1] + bBox[3] + 1, bBox[0]:bBox[0] + bBox[2] + 1]
	if region.shape[0] == 0 or region.shape[1] == 0: #Incase bBox bounds are not in image range or width/height negative
		return
	hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
	hsvValues = getConvertedColorHue(T)
	hsv[:,:,0] = hsvValues[0, 0, 0]
	hsv[:,:,1] = hsvValues[0,  0, 1]
	img[bBox[1]:bBox[1] + bBox[3] + 1, bBox[0]:bBox[0] + bBox[2] + 1] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def coordTransform(coords):
	return (coords[0]) * 4, (coords[1] - 220) * 4, coords[2] * 4, coords[3] * 4, (coords[4]) * 4, (coords[5] - 220) * 4


matplotlib.use('TkAgg')
graph = plt.figure()
graph.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))

plt.ion()
plt.show()


outVideo = cv2.VideoWriter("CompiledVideo.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 10, (2000, 1200))

with open('extractedData.pickle', 'rb') as f:
	minWavelength, maxWavelength, particles = pickle.load(f)

	video = VideoSource.VideoSource(filename="Al3Zr_SM_30k_Run2.avi", skip=0, end=-1, spectraStart=150, spectraEnd=1024 - 1)

	while True:
		frame, frameNum = video.getFrame()
		if frame is None:
			print("Video Done")
			break

		resized = cv2.resize(frame[220:520, 0:500], (2000, 1200)) #X, Y
		resized = decorateFrame(particles, resized, frameNum, showDebugInfo=True, showPath=False, fullBoundingBox=False)

		makeTempGraph(particles, frameNum)

		img = np.array(graph.canvas.buffer_rgba())
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

		resized[:img.shape[0], -img.shape[1]:] = img

		makeSpectraGraph(particles, frameNum, minWavelength, maxWavelength)

		img = np.array(graph.canvas.buffer_rgba())
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

		resized[-img.shape[0]:, -img.shape[1]:] = img


		if frameNum % 50 == 0:
			print(frameNum)
		outVideo.write(resized)
		'''
		cv2.imshow('Data', resized)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		'''
video.destroy()
cv2.destroyAllWindows()
outVideo.release()

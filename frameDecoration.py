import cv2
import colorID

class FrameDecorator:
	def __init__(self):
		self.traces = dict()

	def addToPath(self, pId, x, y, occluded):
		if pId in self.traces:
			self.traces[pId].append((x, y, occluded))
		else:
			self.traces[pId] = [(x, y, occluded)]

	def decorateFrame(self, f, track, frameNum, coordTransform, showDebugInfo=True, showOccluded=True, showPath=True):
		boundingData = track.getTrackingData()
		for particleId, coords, occCount in boundingData:
			x, y, w, h, cX, cY = coordTransform(coords)

			if showOccluded:
				if occCount > 0:
					cv2.putText(f, str(occCount), (x - 15, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA, False)
			else:
				if occCount > 0:
					continue
			cv2.rectangle(f, (x, y), (x + w, y + h), colorID.getColorOfId(particleId) if occCount == 0 else (0, 0, 255), 1)
			cv2.putText(f, str(particleId), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if occCount == 0 else (0, 0, 255), 1, cv2.LINE_AA, False)

			if showPath:
				self.addToPath(particleId, int(cX), int(cY), occCount > 0)

		if showDebugInfo:
			cv2.putText(f, "Frame Num: " + str(frameNum), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
			cv2.putText(f, "Last ID used: " + str(track.getPreviouslyUsedId()), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
			cv2.putText(f, "Particles: " + str(len(boundingData)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

		if showPath:
			for pId in self.traces:
				path = self.traces[pId]
				for (x, y, occluded) in path:
					if not showOccluded:
						assert(occluded == False)
					cv2.circle(f, (x, y), 2, (0, 0, 255) if occluded else colorID.getColorOfId(pId), -1)
				for i in range(len(path) - 1):
					cv2.line(f, (path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1]), colorID.getColorOfId(pId), 1)

			# Delete old paths
			if len(boundingData) == 0:
				for pId in list(self.traces):
					del self.traces[pId]
			else:
				for pId in list(self.traces): #Delete old paths
					if pId not in list(zip(*boundingData))[0]:
						del self.traces[pId]

		return f

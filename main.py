import cv2
import tracking
import colorId

def decorateFrame(f, data, num, prevId):
	cv2.putText(f, "Frame Num: " + str(num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
	cv2.putText(f, "Last ID used: " + str(prevId), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

	for particleId, (x, y, w, h, cX, cY) in data:
		cv2.rectangle(f, (x, y), (x + w, y + h), colors.getColorOfId(particleId), 1)
		#cv2.putText(f, str(particleId), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA, False)

	# cv2.circle(img, (int(cX), int(cY)), 3, (0, 0, 255), -1)
	return f


video = cv2.VideoCapture("MF_AlZr.avi")
frameNum = 0

tracker = tracking.MultiObjectTracker()
colors = colorId.ColorId()

for i in range(250):
	ret, frame = video.read()

while True:
	ret, frame = video.read()

	if not ret:
		print("Bad read")
		break

	cv2.imshow('source', frame)
	trackingData = tracker.processImage(frame)
	cv2.imshow('frame', decorateFrame(frame, trackingData, frameNum, tracker.getPreviouslyUsedId()))

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
	frameNum += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
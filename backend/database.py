from backend import particle


idsInPreviousFrame = set()
particles = dict()
previouslyUsedId = -1

def getNewId():
	global previouslyUsedId
	previouslyUsedId += 1
	return previouslyUsedId

def getPreviouslyUsedId():
	return previouslyUsedId

def createNewParticle(boundingBox, brightness, frameNum):
	newId = getNewId()
	particles[newId] = particle.Particle(newId, boundingBox, brightness, frameNum)
	idsInPreviousFrame.add(newId)

def getParticleById(pId):
	return particles[pId]

def getIdsOfPreviousFrame():
	return list(idsInPreviousFrame)

def assignNewData(pId, bBox, brightness):
	getParticleById(pId).updateBBox(bBox, brightness)
	assert(pId in idsInPreviousFrame)

def isOccluded(pId):
	oldOcclusionCount = particles[pId].getPreviousOcclusionCount()
	if oldOcclusionCount == 3:
		idsInPreviousFrame.remove(pId)
	else:
		getParticleById(pId).propagateFromPrediction()

def getLastBoundingBoxes():
	data = []
	for pId in idsInPreviousFrame:
		data.append((pId, particles[pId].getPreviousBoundingBox(), particles[pId].getPreviousOcclusionCount()))
	return data

def addSpectraData(spectraDict):
	for pId in spectraDict:
		getParticleById(pId).addSpectraData(spectraDict[pId])

def getFullDataForPickling(videoWidth):
	for pId in particles:
		particles[pId].prepareForPickling(videoWidth)
	return particles
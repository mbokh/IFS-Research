import pickle
import csv


outFile = open('extractedData/dataRestrictedRange.csv', 'w')
writer = csv.writer(outFile, lineterminator = '\n')
#writer.writerow(["// Format: pId, frame #, [temperature, [spectra], spectra code], (bounding box), brightness, occlusion count"])

with open('extractedData/extractedDataRestrictedRange.pickle', 'rb') as f:
	minWavelength, maxWavelength, particles = pickle.load(f)

	writer.writerow([minWavelength, maxWavelength])

	for pId in sorted(particles.keys()):
		p = particles[pId]
		for i in range(len(p.spectraData)):
			writer.writerow([pId,
							 p.frameNumAppeared + i,
							 p.spectraData[i][0],
							 p.spectraData[i][1],
							 p.spectraData[i][2],
						  	 p.particleData[i][0],
							 p.particleData[i][1],
							 p.particleData[i][2]])

outFile.close()
import pickle
import csv


outFile = open('data.csv', 'w')
writer = csv.writer(outFile, lineterminator = '\n')
#writer.writerow(["// Format: pId, frame #, [temperature, [spectra], spectra code], (bounding box), brightness, occlusion count"])

with open('extractedDataMedianBlur.pickle', 'rb') as f:
	minWavelength, maxWavelength, particles = pickle.load(f)

	writer.writerow([minWavelength, maxWavelength])

	for pId in sorted(particles.keys()):
		p = particles[pId]
		for i in range(len(p.spectraData)):
			writer.writerow([pId,
							 p.frameNumAppeared + i,
							 p.spectraData[i],
						  	 p.particleData[i]])

outFile.close()
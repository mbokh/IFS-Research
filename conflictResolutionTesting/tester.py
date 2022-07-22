import numpy as np
import leastSquares
import bruteForce
import utils
import time
from matplotlib import pyplot as plt

temps = [2350, 2750, 2350, 2520]
offsets = [0, 30, 60, 90]

utils.responseFunction = utils.gaussianResponse

curves = [utils.createCurve(temps[i], offsets[i], offsets[-1]) for i in range(len(temps))]
combinedIntensities = sum(curves)

combinedX = np.linspace(0, utils.pixelEnd + offsets[-1] - 1, utils.pixelEnd + offsets[-1])
fig, axs = plt.subplots(2, 1)
for c in curves:
	axs[0].plot(combinedX, c)
axs[1].plot(combinedX, combinedIntensities, 'g')
fig.tight_layout()
plt.show()

print("Least Squares:")
t = time.time_ns()
solution1 = leastSquares.optimize(offsets, combinedIntensities)
print((time.time_ns() - t) / 1000000000)
for i in range(len(temps)):
	print(int(solution1[i]))


print()
print()
print("Brute Force:")
t = time.time_ns()
solution2 = bruteForce.optimize(offsets, combinedIntensities)
print((time.time_ns() - t) / 1000000000)
for i in range(len(temps)):
	print(int(solution2[i]))

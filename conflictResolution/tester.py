import numpy as np
import leastSquares
import bruteForce
import utils
import time
from matplotlib import pyplot as plt

temps = [2323, 2650, 2400]
gains = [0.3, 0.73, 0.91]
offsets = [0, 240, 310]

utils.responseFunction = utils.gaussianResponse

curves = [utils.createCurve(temps[i], gains[i], offsets[i], offsets[-1]) for i in range(len(temps))]
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
	print(str(int(solution1[2 * i])) +  ", " + str(np.format_float_positional(solution1[2 * i + 1], precision=3)))


print()
print()
print("Brute Force:")
t = time.time_ns()
solution2 = bruteForce.optimize(offsets, combinedIntensities)
print((time.time_ns() - t) / 1000000000)
for i in range(len(temps)):
	print(str(int(solution2[2 * i])) + ", " + str(np.format_float_positional(solution2[2 * i + 1], precision=3)))

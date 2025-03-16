import numpy as np
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

np.random.seed(0)

n = 5
points = np.random.default_rng(1).choice(n * 2 + 1, (n, 2), replace=False)
points = np.array(sorted(points, key=lambda x: x[0]))


def naturalCubicSplines(points, edgeType="natural"):
	n = points.shape[0]
	mat = np.zeros((4 * (n - 1), 4 * (n - 1)))

	for i in range(1, n):
		x1 = [points[i - 1][0] ** j for j in range(3, -1, -1)]
		x2 = [points[i][0] ** j for j in range(3, -1, -1)]
		mat[(i - 1), 4 * (i - 1):(4 * (i - 1) + 4)] = x1
		mat[(n - 2) + i, 4 * (i - 1):(4 * (i - 1) + 4)] = x2

	for i in range(n - 2):
		x1 = np.multiply([points[i + 1][0] ** j if j >= 0 else 0 for j in range(2, -2, -1)], [3, 2, 1, 0])
		x2 = np.multiply([points[i + 1][0] ** j if j >= 0 else 0 for j in range(1, -3, -1)], [6, 2, 0, 0])
		mat[2 * (n - 1) + i, 4 * i:(4 * i + 8)] = np.append(x1, -x1)
		mat[2 * (n - 1) + (n - 2) + i, 4 * i:(4 * i + 8)] = np.append(x2, -x2)
	if edgeType.lower() == "natural":

		# this is for natural cubic splines
		for i in range(2):
			x = np.multiply([points[i][0] ** j if j >= 0 else 0 for j in range(1, -3, -1)], [6, 2, 0, 0])
			mat[2 * (n - 1) + 2 * (n - 2) + i, 4 * (n - 2) * i:4 * (n - 2) * i + 4] = x
	elif edgeType.lower() == "clamped":
		# this is for clamped cubic splines
		for i in range(2):
			x = np.multiply([points[i][0] ** j if j >= 0 else 0 for j in range(2, -2, -1)], [3, 2, 1, 0])
			mat[2 * (n - 1) + 2 * (n - 2) + i, 4 * (n - 2) * i:4 * (n - 2) * i + 4] = x
	else:
		raise "Enter Valid Edge Type (clamped or natural)"
	y = np.append(points[:n - 1][:, 1], points[1:][:, 1])
	y = np.append(y, np.zeros(mat.shape[0] // 2))
	y[-2] = 3
	y[-1] = 0
	y = y[:, np.newaxis]

	return mat, y


mat, y = naturalCubicSplines(points, 'clamped')

coefficients = np.dot(np.linalg.inv(mat), y)


def cubic(coefs):
	return lambda x: np.sum(np.multiply(coefs, [x ** 3, x ** 2, x ** 1, x ** 0]), axis=0)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), layout='tight')  # type:figure.Figure, axes.Axes

for i in range(points.shape[0]):
	ax[0].scatter(points[i][0], points[i][1])

for i in range(points.shape[0] - 1):
	x = np.linspace(points[i, 0], points[i + 1, 0], 100)
	y = coefficients[4 * i:4 * i + 4]
	ax[0].plot(x, cubic(y)(x))

poly = lagrange(points[:, 0], points[:, 1])

x_new = np.arange(points[0, 0], points[-1, 0], 0.001)
ax[1].scatter(points[:, 0], points[:, 1])
ax[1].plot(x_new, np.polynomial.Polynomial(poly.coef[::-1])(x_new))

plt.show()

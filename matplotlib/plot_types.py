# different types of charts in matplotlib

# line plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.plot(x, y)
# plt.show()

# scatter plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.scatter(x, y)
# plt.show()

# bar plot
# x = [1, 2, 3, 4, 5,6,7,8,9,10]
# y = [10, 20, 30, 40, 50,60,70,80,90,100]
# plt.bar(x, y)
# plt.show()

# histogram
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# Make data
n = 20
x = np.sin(np.linspace(0, 2*np.pi, n))
y = np.cos(np.linspace(0, 2*np.pi, n))
z = np.linspace(0, 1, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.stem(x, y, z)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()



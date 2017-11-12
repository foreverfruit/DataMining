import matplotlib.pyplot as plt

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot([1,2],[1,2])

f2 = plt.figure()
ax2 = f2.add_subplot(222)
ax2.plot([1,2],[2,1])

plt.show()
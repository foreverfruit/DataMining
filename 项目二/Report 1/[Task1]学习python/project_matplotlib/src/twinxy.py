import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,11)

y1 = x**2
y2 = -3*x+3

fig = plt.figure()
ax = fig.add_subplot(111)

l1, = ax.plot(x,y1,color='r',label='$y1=x^2$')

ax2 = ax.twinx()
l2, = ax2.plot(x,y2,color='b',label='$y2=-3*x+3$')

# 这里需要了解axe和figure的范围，若gcf()再legend，图例会滑到整个图的四个角，坐标轴外面
# 若gca()表示获取到轴，再legend，图例画在轴范围内
plt.gca().legend((l1,l2),('A','B'),loc=0)
plt.show()


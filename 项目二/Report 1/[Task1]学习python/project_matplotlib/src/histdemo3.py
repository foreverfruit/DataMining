# 二维的直方图，用颜色表示联合密度
import numpy as np
import matplotlib.pyplot as plt

# xy均值分别为1和5，方差都为1
x = 1+np.random.randn(2000)
y = 5+np.random.randn(2000)

plt.hist2d(x,y,bins=40)
# 图通过颜色深浅表示概率，显然在(x,y)=(1,5)处密度最大，颜色最亮
plt.show()
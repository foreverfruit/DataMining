import  numpy as np
import  matplotlib.pyplot as plt

N = 100

xlist = np.random.randn(N)
ylist = np.random.randn(N)
# 这里 y和x 是负相关
ylist = xlist*(-10) + np.random.randn(N)

plt.scatter(xlist,ylist,marker='*')

plt.show()
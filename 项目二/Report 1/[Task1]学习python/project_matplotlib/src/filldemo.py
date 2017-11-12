# ipython file 用于交互式输入输出
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5*np.pi,1000)

y1 = np.sin(x)
y2 = np.cos(x)
'''

plt.fill(x,y1,'b',alpha=0.3)
plt.fill(x,y2,'r',alpha=0.3)

'''
# 交集填充，where表示填充区域的条件表达式
# interpolate表示是否完全填充，因为数据点是离散的，粒度太粗的时候，会有区域填充不到，这个参数会实现完全填充
plt.fill_between(x,y1,y2,where=(y1>y2),facecolor='r',alpha=0.5,interpolate=True)
plt.fill_between(x,y1,y2,where=(y1<y2),facecolor='b',alpha=0.5)

plt.show()
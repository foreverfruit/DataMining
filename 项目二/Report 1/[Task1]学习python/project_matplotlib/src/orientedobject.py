# 面向对象的方式画图
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,5)
y = np.random.randn(len(x))

fig = plt.figure()
ax = fig.add_subplot(224)
l,=plt.plot(x,y)
t = ax.set_title('hehe')
plt.show()
# 随机生成数据做直方图
import numpy as np
import matplotlib.pyplot as plt

mean = 20
var = 0.5
data = mean+var*np.random.randn(100000)

# plt.hist(data,bins=50,normed=False,color='red')
# 当直方图的划分很大时，就是密度曲线
plt.hist(data,bins=500,normed=True,color='red')
plt.show()
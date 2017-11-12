"""
累计条形图，stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import utils as tool

sepal_length,sepal_width,petal_length,petal_width = np.loadtxt(tool.getPath()+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))

# data
sun_sepal_length = [np.sum(sepal_length[0:50]),np.sum(sepal_length[51:100]),np.sum(sepal_length[101:150])]
sun_sepal_width = [np.sum(sepal_width[0:50]),np.sum(sepal_width[51:100]),np.sum(sepal_width[101:150])]
sun_petal_length = [np.sum(petal_length[0:50]),np.sum(petal_length[51:100]),np.sum(petal_length[101:150])]
sun_petal_width = [np.sum(petal_width[0:50]),np.sum(petal_width[51:100]),np.sum(petal_width[101:150])]


# plot
left = np.arange(3)
width = 0.5
# 这里有大量重复的矩阵计算，需要优化
bar1 = plt.bar(left,sun_sepal_length,width,yerr=sun_sepal_length,align='center',color='red')
bar2 = plt.bar(left,sun_sepal_width,width,yerr=sun_sepal_width,bottom=sun_sepal_length,align='center',color='green')
bar3 = plt.bar(left,sun_petal_length,width,yerr=sun_petal_length,bottom=np.add(sun_sepal_length,sun_sepal_width),align='center',color='blue')
bar4 = plt.bar(left,sun_petal_width,width,yerr=sun_petal_width,bottom=np.add(np.add(sun_sepal_length,sun_sepal_width),sun_petal_length),align='center',color='black')
# 坐标轴美化
plt.ylabel('value')
plt.title('this is title')
plt.xticks(left,('typeA','typeB','typeC'))
# 图例标注
plt.legend((bar1[0], bar2[0],bar3[0],bar4[0]), ('sepal_length', 'sepal_width','petal_length','petal_width'))
plt.show()
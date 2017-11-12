"""
并列对比条形图
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

x = np.arange(0,15,5)
width = 0.5

bar1 = plt.bar(x,sun_sepal_length,width,color='red')
bar2 = plt.bar(x+width+0.1,sun_sepal_width,width,color='green')
bar3 = plt.bar(x+width*2+0.2,sun_petal_length,width,color='blue')
bar4 = plt.bar(x+width*3+0.3,sun_petal_width,width,color='black')

plt.xticks(x+width*2,('A','B','C'))
plt.legend((bar1[0],bar2[0],bar3[0],bar4[0]),('sepal_length','sepal_width','petal_length','petal_width'),loc=2)

plt.show()
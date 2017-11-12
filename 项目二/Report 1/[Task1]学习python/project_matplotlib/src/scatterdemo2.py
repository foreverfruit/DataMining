# 查看花的sepal的length和width的关系
import  numpy as np
import  matplotlib.pyplot as plt
import  os
pro_root = os.path.abspath('..')

sepal_length,sepal_width = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1))

plt.scatter(sepal_width[:50],sepal_length[:50],marker='*',c='red',alpha=0.5,s=50)
plt.scatter(sepal_width[51:100],sepal_length[51:100],marker=',',c='green',alpha=0.5,s=50)
plt.scatter(sepal_width[101:150],sepal_length[101:150],marker='.',c='blue',alpha=0.5,s=50)
plt.show()

# 可以看到三种花的sepal的宽和长都是近似正相关的
import  numpy as np
import  matplotlib.pyplot as plt
import  os
pro_root = os.path.abspath('..')
data = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=False,dtype=float,usecols=(0,1,2,3))

for x in data[0:50]:
    plt.plot(x,'r--')
for x in data[51:100]:
    plt.plot(x,'g-')
for x in data[100:150]:
    plt.plot(x,'b-.')

# 对图做个加工
# x、y轴标签与图形标题
plt.xlabel('category')
plt.ylabel('value')
plt.title('iris\'s sepal and petal leagth and width')

# X轴上的标签
xticks=['sepal length','sepal width','petal length','petal width']
#设置x轴的刻度，将构建的xticks代入
plt.xticks(range(len(xticks)),xticks,rotation=0)

# 图中可以看出三种类型的折线图
plt.show()

[TOC]

---

# Project 2 _ Report 1

本课程相关内容已划分好文件夹全部上传到Github，Clone地址如下：

[https://github.com/foreverfruit/DataMining.git](https://github.com/foreverfruit/DataMining.git)

---

## 计划任务

根据本课程项目二计划书，第一阶段任务如下（摘自proposal）：

- 数据预处理，将文本数据整理成需要的格式。
- 完成python语言的学习。
- 学习一种分类算法的理论，尝试采用python实现，或者至少能应用已有开源库做一些该算法的demo。 


---


## Task1_数据预处理

---

## Task2_Python学习

本阶段学习内容为python语言的语法学习，以及常用的科学计算库matplotlib和numpy的简单使用，能使用python语言实现数据处理、计算和可视化任务。

具体的学习笔记见本项目下文件：

- `/项目二/[Task1]学习python/python.md`
- `/项目二/[Task1]学习python/matplotlib & numpy.md`
- matplotlib练习代码：`/项目二/[Task1]学习python/project_matplotlib`

github地址：[Task 2：python学习](https://github.com/foreverfruit/DataMining/tree/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask1%5D%E5%AD%A6%E4%B9%A0python)



## Task3_决策树实现

这部分的学习内容主要是完成一个小的练习项目，对鸢尾花数据集进行挖掘，采用决策树的算法实现预测分类功能。

数据集：[http://archive.ics.uci.edu/ml/datasets/Iris](http://archive.ics.uci.edu/ml/datasets/Iris)

***数据集信息***

- 数据量小：150个记录，每个类50，共3个类标号
- 属性：4维，连续型
- 数据完整，不存在缺失项


***数据可视化***

这里可视化采用的是原始数据，为了大致观测到数据的一些

1.各属性直方图

![pic](https://raw.githubusercontent.com/foreverfruit/DataMining/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/pics/attr_hist.png)

可以看到，属性petal_length和属性petal_width对类的区分度非常高，也和统计数据中的相关度相吻合。（事实上，下文的决策树在结点分裂的时候，会选用划分后Gini指数最低的属性进行划分，即子集的纯度最高，显然第一次的时候用属性petal_width会得到效果最好的划分。）

```python
import  numpy as np
import  matplotlib.pyplot as plt
import  os

# load data
pro_root = os.path.abspath('..')
sl,sw,pl,pw = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))
datarray = np.array([sl,sw,pl,pw])
attr_names = ['sepal length','sepal width','petal length','petal width']
class_names = ['setosa','versicolor','virginica']

# plot
plt.style.use('ggplot')
figure = plt.figure()
for i,data in enumerate(datarray):
    d1,d2,d3 = data[:50],data[50:100],data[100:]
    axe = figure.add_subplot(221+i)
    axe.hist(d1,color='r',bins=10,alpha=0.5)
    axe.hist(d2,color='g',bins=10,alpha=0.5)
    axe.hist(d3,color='b',bins=10,alpha=0.5)
    axe.set_title(attr_names[i])
    axe.legend(labels = class_names)

plt.show()
```



2.各属性盒装图

这里取whis参数为0.9，用以控制离群点的“离群”程度定义，图中可以看出类别1在4种属性上的分布都比较分散。

![pic2](https://raw.githubusercontent.com/foreverfruit/DataMining/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/pics/attr_box.png)

```python
import  numpy as np
import  matplotlib.pyplot as plt
import  os

# load data
pro_root = os.path.abspath('..')
sl,sw,pl,pw = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))
datarray = np.array([sl,sw,pl,pw])
attr_names = ['sepal length','sepal width','petal length','petal width']
class_names = ['setosa','versicolor','virginica']

# plot
plt.style.use('ggplot')
figure = plt.figure()
for i,data in enumerate(datarray):
    d1,d2,d3 = data[:50],data[50:100],data[100:]
    axe = figure.add_subplot(221+i)
    axe.boxplot((d1,d2,d3), whis=0.9, sym='rx')
    axe.set_title(attr_names[i])
    axe.set_xticklabels(class_names)

plt.show()
```



3、三维散布图

这里选取相关度最高的三个属性做三维图，可以看到红色的类型的区别度较高，也印证直方图的属性分布图中，红色的类型setosa的4个属性与其他类型相应的属性取值范围间隔较大，几乎没有相交的，反映到三维途中会有直观的“距离”感

![pic3](https://raw.githubusercontent.com/foreverfruit/DataMining/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/pics/attr_3d_scatter.png)

```python
import  numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  os

# load data
pro_root = os.path.abspath('..')
sl,sw,pl,pw = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))
datarray = np.array([sl,sw,pl,pw])
attr_names = ['sepal length','sepal width','petal length','petal width']
class_names = ['setosa','versicolor','virginica']

# plot
plt.style.use('ggplot')
figure = plt.figure()
axe = figure.add_subplot(111, projection = '3d')

axe.scatter(sw[:50], pl[:50], pw[:50], c='r', marker='^')
axe.scatter(sw[50:100], pl[50:100], pw[50:100], c='g', marker='x')
axe.scatter(sw[100:150], pl[100:150], pw[100:150], c='b', marker='.')

axe.set_xlabel(attr_names[1])
axe.set_ylabel(attr_names[2])
axe.set_zlabel(attr_names[3])

plt.show()
```



***数据预处理***

这里数据集较为简单，没有做过多数据预处理，仅在加载数据的同时，就地做了离散化工作，根据统计信息（如下表），做出相应的离散映射

| Class        | Min  | Max  | Mean | SD   | Class Correlation |
| ------------ | ---- | ---- | ---- | ---- | ----------------- |
| sepal length | 4.3  | 7.9  | 5.84 | 0.83 | 0.78              |
| sepal width  | 2.0  | 4.4  | 3.05 | 0.43 | -0.4194           |
| petal length | 1.0  | 6.9  | 3.76 | 1.76 | 0.9490            |
| petal width  | 0.1  | 2.5  | 1.2  | 0.76 | 0.9565            |

映射步骤：根据每个属性的最大值最小值，取得一个取值空间，然后以0.5为一个步长，将这个取值空间映射到从1开始的整数序列上，如petal width取值范围(0.1,2.5)，则取范围[0,3]根据0.5步长映射为，1 = [0,0.5),2 = [0.5,1),3 = [1,1.5), 4 = [1.5,2), 5 = [2,2.5]，实现代码如下：

```python
newset = []
# 离散值集合
valueset=range(1,13)
# 遍历处理每一条记录
for record in dataset:
    newrecord = []
    for index,value in enumerate(record):
        i = 0
        if index==0:
            # sepal length[4.3,7.9]取[4,8]内按0.5步长做区间划分
            i = int((float(value)*10-40)/5)
        if index==1:
            i = int((float(value) * 10 - 20) / 5)
        if index==2:
            i = int((float(value) * 10 - 10) / 5)
        if index==3:
            i = int(float(value)*10/5)
        # class label 属性
        if index==4:
            newrecord.append(classlabel.get(value))
        else:
            newrecord.append(valueset[i])
	newset.append(newrecord)
```



***决策树实现***

这里直接采用书中决策树的描述，根据iris数据集的特点实现了决策树，实现过程中主要有一下重点问题。

- 结点分裂算法：Hunt算法，将数据当做一个数据流，边构造决策树边流动，直到所有的数据都流到了叶结点上。递归执行划分方法。

- 划分问题1：属性划分点的确定，书中着重介绍了二元划分的方案，本数据集是三元划分。三元划分可以转换为二元划分，对于某一特定的属性A，对它的取值空间做二元划分，用划分后的子集的纯度（这里我采用Gini指数度量）来衡量选取属性的效果。这种划分方法优点是实现简单，但会造成决策树深度过大。

  我采用的方法：多路划分，根据划分评价的原理：寻找子类纯度最高的划分。那么可以认为最优的划分就是经过这次划分后，子类中类别区分最明显，这里根据数据集特点做出假设：每种类别的记录在某一个属性上的取值明显呈现“聚堆”现象。那么最优的划分点就是这些“堆”的边界点。本例中划分点没有通过代码计算，采用图像观测。图像如下：

  ![pic4](https://raw.githubusercontent.com/foreverfruit/DataMining/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/pics/divide_point1.png)

  ![pic5](https://raw.githubusercontent.com/foreverfruit/DataMining/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/pics/divide_point2.png)

  上面的离散处理之后的图像，下面的是未处理的原始数据图像。根据图像可以找到如下的划分点：（原始图上值和划分点值有差距，是因为离散化映射的问题）

  ```python
  # 因为值是int的，这里划分点取float，避免值等于划分点时的区间开闭情况
  dp = {'sl':(3.5,5.5),'sw':(1.5,2.5,3.5,4.5),'pl':(3,),'pw':(2.1,3.8,4.8)}
  ```

  这里取得上面的各属性的划分点就可以近似认为是最优的划分点，Gini最小，子类的纯度最高。

- 划分问题2：中止划分条件。1）当前结点的所有数据元素的属性值都相同，此时若类标号不同，取多数的类标号，少数的认为是异常值。2）当前结点95%的元素都是同一个类标号。

- 划分问题3：属性选取策略，一个结点具有n个可划分的属性（已经划分过的属性在该结点中元素取值肯定是相同的），问题是先选用n个中的哪一个属性作为当前的划分属性，此时已知各属性的最优划分点，所以可以用已知的划分点，对该结点的数据模拟各属性上的划分（模拟n次），计算这n次划分后的子集合的**加权Gini系数**，然后选取Gini最小的那个属性作为该结点划分属性。

- 划分问题4：可能存在一些结点，根据属性的划分点做划分后，并没有一条数据流入该子结点，此时，同样要为决策树创建这样的空叶结点，并做好标记，因为训练集没有数据流入该结点，测试集可能存在，要避免测试集数据找不到叶结点的情况，它的类别取其父节点中最多的类别。

- 算法流程：创建RootNode(这个TrainingSet流入该结点)，判断当前结点能否分裂（两个中止条件都要判定），选取最优的属性做划分（属性选取算法加权Gini），为每一个划分后的子数据集创建结点（数据分散流入子节点），递归调用子节点的划分函数divide()

***算法评估***

时间原因，没有做深入评估，仅做了正确率的验证：90%左右（交换不同训练集和测试集得到的近似结果，没做具体的统计）

***不足之处***

- 模型评估未实现
- 正确率在小数据量的情况下不高
- 由于数据集较小，没有做决策树剪枝，拟合分析
- 数据预处理粗糙
- 实现过程中简单起见，没有设置过多参数以增加代码的通用性，如数据集拆分比率、属性划分点、划分阈值都是定值。

完整项目代码：[Iris_DecisionTree_Github](https://github.com/foreverfruit/DataMining/tree/master/%E9%A1%B9%E7%9B%AE%E4%BA%8C/Report%201/%5BTask3%5D%E5%AE%9E%E7%8E%B0%E4%B8%80%E7%A7%8D%E5%88%86%E7%B1%BB%E6%96%B9%E6%B3%95-%E5%86%B3%E7%AD%96%E6%A0%91/iris)


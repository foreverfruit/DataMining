# Project 2 _ Report 1



## 计划任务

- 数据预处理，将文本数据整理成需要的格式。
- 完成python语言的学习。
- 学习一种分类算法的理论，尝试采用python实现，或者至少能应用已有开源库做一些该算法的demo。 



## Task 1_数据预处理



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

- 给出的统计：

  | Class        | Min  | Max  | Mean | SD   | Class Correlation |
  | ------------ | ---- | ---- | ---- | ---- | ----------------- |
  | sepal length | 4.3  | 7.9  | 5.84 | 0.83 | 0.78              |
  | sepal width  | 2.0  | 4.4  | 3.05 | 0.43 | -0.4194           |
  | petal length | 1.0  | 6.9  | 3.76 | 1.76 | 0.9490            |
  | petal width  | 0.1  | 2.5  | 1.2  | 0.76 | 0.9565            |




***数据可视化***

这里可视化采用的是原始数据，为了大致观测到数据的一些

1.各属性散点图

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


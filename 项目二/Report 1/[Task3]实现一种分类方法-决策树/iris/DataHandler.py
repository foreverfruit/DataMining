# 数据相关内容
import numpy as np
import matplotlib.pyplot as plt

# 这里类标号直接根据数据集写死了，不再设置通用的检测函数，同理属性集也一样
classlabel = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
attrlabel = ('sepal length','sepal width','petal length','petal width')

def loadData():
    """
    加载数据,返回训练集和测试集
    """
    trainingset,testset,dataset = [],[],[]

    with open('iris.data','r') as f:
        data = f.readlines()
        for i,e in enumerate(data):
            # 去掉行末的换行符
            e = e.strip('\n')
            # 将数据分割成列表
            e = e.split(',')
            # 2:1的比例分发数据到训练集和测试集，即分别100个和50个数据
            dataset.append(e)
            if(i%3):
                trainingset.append(e)
            else:
                testset.append(e)

    return trainingset,testset,dataset



"""
数据统计信息：
Summary Statistics:
	         	 Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826   
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
"""
def discretization(dataset:list):
    """
    数据预处理：数据离散化，这里针对数据集做两个离散处理
    1、四个一般属性值根据取值的粒度(这里划分步长都取0.5)进行离散化划分，映射到整数域中
    2、对类标号做映射
    """
    # 新数据集，离散化后的数据集
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

            if index==4:
                newrecord.append(classlabel.get(value))
            else:
                newrecord.append(valueset[i])
        newset.append(newrecord)
    return newset


def dividepoints(dataset:list):
    """
    根据数据集，为数据集的每个属性计算划分点,
    这里划分点是根据整个数据集固定的，并不会在每次树分裂的时候动态调整。(也不能动态调整，不然一条数据的时候怎么做预测)
    划分点算法：一种先验的方式寻找多路划分点，假设每个属性都能区分所有的类别（这在本数据集是成立的，但不普遍适用）
    故根据分布密度，找到n-1个分布边界点，n为类别总数，本数据集类别n=3，故每种属性找到两个划分点
    输入：离散化后的测试集
    输出：每个属性对应的两个划分点
    实际情况有调整：原计划固定3路划分，作图后发现，不理想，故按照不定的多路划分
    """

    '''
    这里没有代码实现，直接采用了作图观测值，需要后续改进，代码如下
    
    train, test, total = loadData()
    train2 = discretization(train)
    arr = np.array(train2)

    figure = plt.figure()

    for i in range(4):
        a = arr[:, i]
        axe = figure.add_subplot(220+i+1)
        axe.hist(a,bins=10,normed=True)
        axe.set_title(attrlabel[i])

    plt.show()
    
    '''
    # 因为值是int的，这里划分点取float，避免等于划分点的情况
    dp = {'sl':(3.5,5.5),'sw':(1.5,2.5,3.5,4.5),'pl':(3,),'pw':(2.1,3.8,4.8)}
    return dp

if __name__ == '__main__':
    train, test, total = loadData()
    train2 = discretization(train)
    arr = np.array(train2)

    figure = plt.figure()

    for i in range(4):
        a = arr[:, i]
        print(a)
        axe = figure.add_subplot(220 + i + 1)
        axe.hist(a,normed=True)
        axe.set_title(attrlabel[i])

    plt.show()
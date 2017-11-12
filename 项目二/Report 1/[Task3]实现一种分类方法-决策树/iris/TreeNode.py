# 树节点
import numpy as np
import DataHandler as dh

class TreeNode(object):
    """
    树节点
    """
    # DataHandler中得到的 divide points
    dp = {'sl': (3.5, 5.5),
          'sw': (1.5, 2.5, 3.5, 4.5),
          'pl': (3,),
          'pw': (2.1, 3.8, 4.8)}

    def __init__(self,dataset:list):
        # 本结点的子节点
        self.children=[]
        # 本节点的数据集合
        self.data = dataset
        # 当前结点类型为数据集合中多的那一类的类标号
        self.classlabel=''
        # 结点类型，0为非叶结点，1为叶结点
        self.nodetype = -1
        # 该结点的划分属性,sl\sw\pl\pw 分别为 0,1,2,3,叶结点不再分裂，值为-1
        self.divideAttr=-1



    def insert(self,node):
        """插入子结点"""
        self.children.append(node)

    def divideAttr(self):
        """
        根据当前结点的数据集寻找最佳划分的属性
        对比各属性的划分后（根据对应的属性划分点）的Gini指数，选择最佳的划分属性
        :return 返回属性标号
        """


        pass


    def divide(self):
        """
        就是根据属性测试条件，以及测试点，将数据集合分成几个部分，分别建立TreeNode然后加入子节点集合
        返回子节点集合
        """
        pass
        return (1,2,3)

    def checkDivide(self):
        """
        两个划分结束判断：
        1. 数据集合的属性集合相同（离散情况下就是所有实体的每个属性都在相同的划分区间）
        2. 数据集中的90%的实体是同一个类标号
        同时根据该检测，为该结点设置属性（叶、非叶）
        :returns 可作为划分的属性集合，或者None表示不可再划分
        """
        array = np.array(self.data)
        result = []

        # 属性取值判断，所有属性都在同一个划分区间，则该点不能再分裂
        for i in range(4):
            # 第i个属性的值检测，是否在一个划分区间内，是，则表示该属性不能再做划分条件
            column = array[:, i]
            min_v,max_v = column.min(),column.max()
            keys = list(self.dp.keys())
            points = self.dp.get(keys[i])
            for p in points:
                if min_v<p and max_v>p:
                    # 存在这样一个属性的划分点可以进行划分，则缓存该属性标号
                    result.append(i)
                    break

        # 类标号判定
        classes = list(array[:, 4])
        count_of_class = [classes.count(s) for s in set(classes)]
        #print(type(classes),type(count_of_class))
        flag_class_can_divide = False if max(count_of_class)/len(classes)>0.9 else True

        # 仅当存在可划分的属性且类标号不同(0.9阈值)时，返回可作为划分的属性的列表，表示可划分
        if len(result)>0 and flag_class_can_divide:
            # 本结点可划分，设置本结点属性
            self.nodetype = 0

            return result
        else:
            return None


if __name__ == '__main__':
    trainingset, testset, dataset = dh.loadData()
    s_trainset = dh.discretization(trainingset)
    s_testset = dh.discretization(testset)
    s_dataset = dh.discretization(dataset)

    root = TreeNode(s_trainset)
    print(root.checkDivide())

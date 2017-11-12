# 树节点
import numpy as np
import DataHandler as dh

class TreeNode(object):
    """
    树节点
    """
    # DataHandler中得到的 divide points,依次为 sl\sw\pl\pw
    dp = [(3.5, 5.5),(1.5, 2.5, 3.5, 4.5),(3,),(2.1, 3.8, 4.8)]

    def __init__(self,dataset:list,gap:int=-1,nodetype:int=0,classlabel:int=0):
        print('------------- Node Create --------------------')
        # 本结点的子节点
        self.children=[]
        # 本节点的数据集合
        self.data = dataset
        # gap：这个参数是由父节点传下来的，表示当前结点作为子节点由父节点的某一个属性划分下来
        # gap指示划分的区间,默认为-1，无意义，如gap=0，表示当前结点是父节点划分时候的第一个划分区间对应的子结点
        self.gap = gap

        # 当前结点类型为数据集合中多的那一类的类标号:'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3
        self.classlabel=classlabel
        # 结点类型，0为非叶结点，1为叶结点，2为空叶结点
        self.nodetype = nodetype
        # 该结点的划分属性,sl\sw\pl\pw 分别为 0,1,2,3,值为-1表示当前结点不能分裂，为叶结点
        self.divideAttr=-1

        # 自检，设置属性
        if nodetype!=2: # 空叶节点不用检查了
            self.checkself()


    def checkself(self):
        """
        自我检测函数，实际就是根据数据集，检测并设置一些属性，主要内容包括：
        1、本节点是否可划分
        2、本结点属于什么类型：叶、非叶
        3、本节点的 class label
        4、本节点的划分属性（前提是可划分）
        """
        divideset = self.checkDivide()
        if(divideset):
            # 可划分
            self.nodetype=0
            self.divideAttr=self.findDivideAttr(divideset)
        else:
            # 不可划分,叶结点
            self.nodetype=1
            self.classlabel=self.findlabel()
            print('---- info leaf node,label:',self.classlabel,'---------')


    def findlabel(self):
        """
        :return: int,返回当前结点的类标号
        """
        array = np.array(self.data)
        classes = list(array[:, 4])
        count_of_class = {s:classes.count(s) for s in set(classes)}

        result = maxcount = -1
        for cls,count in count_of_class.items():
            if count > maxcount:
                maxcount = count
                result = cls

        # info print,当前结点中，属于该结点类标号的数据的占比
        print('info - label ratio:',maxcount/len(classes))
        return result


    def findDivideAttr(self,divideset:list):
        """
        根据当前结点的数据集寻找最佳划分的属性
        计算divideset中属性的划分后（根据对应的属性划分点）的Gini指数，选择最佳的划分属性
        :return 返回属性标号
        """
        npdata = np.array(self.data)
        # 结果集： 属性-Gini 键值对
        ginis = {}
        for attr in divideset:
            print('-------- attr:',attr,'-----------')
            # 尝试做对属性attr的划分，得到该划分下的Gini指数
            dpoints = self.dp[attr]  # dpoints = (1.5, 2.5, 3.5, 4.5) 这种形式的划分点
            print('info: divide points - ',dpoints)
            attrvalues = npdata[:,attr] # 所有记录的该属性值的列表
            amount = attrvalues.size # 数据总数目
            # 各划分区间的数据集
            recordset = []
            for i in range(len(dpoints)+1): # 划分区间比划分点大1
                temp = []
                recordset.append(temp)

            # 遍历每个记录的该属性值，进行划分
            for i,v in enumerate(attrvalues):
                # 当前记录所在的划分区间号
                gap = 0
                for p in dpoints:
                    if v > p:
                        gap += 1
                recordset[gap].append(self.data[i])

            # log print
            print('info divide set:')
            for i in range(len(dpoints) + 1):
                print(np.array(recordset[i]))
                print('----------')
            # end print

            # 统计recordset，得到每个划分区间（子节点）的gini
            gini = 0
            for node in recordset:
                # 记录该结点下三个类别的数量
                c1,c2,c3 = 0,0,0
                for i in node:
                    if i[4]==1:
                        c1+=1
                    elif i[4]==2:
                        c2+=1
                    elif i[4]==3:
                        c3+=1
                # 该子节点记录数
                nodeamount = c1+c2+c3
                if nodeamount==0:
                    continue
                # 该节点带权gini指数
                g = nodeamount/amount * ( 1 - (c1/nodeamount)**2 - (c2/nodeamount)**2 - (c3/nodeamount)**2 )
                # 所有字节点带权gini指数和
                gini = gini + g

            # 保存该属性划分的Gini系数
            ginis.__setitem__(attr,gini)

        # info print
        print('info ginis:',ginis)

        # 找最小的gini，返回其属性标号，作为该结点的属性划分
        min_gini = min(ginis.values())
        result = -1
        for at,gi in ginis.items():
            if gi==min_gini:
                result = at
                break

        print('info divide attr:',result,min_gini)

        return result


    def checkDivide(self):
        """
        两个划分结束判断：
        1. 数据集合的属性集合相同（离散情况下就是所有实体的每个属性都在相同的划分区间）
        2. 数据集中的90%的实体是同一个类标号
        同时根据该检测，为该结点设置属性（叶、非叶）
        :returns 可作为划分的属性集合，或者False表示不可再划分
        """
        array = np.array(self.data)
        result = []

        # 属性取值判断，所有属性都在同一个划分区间，则该点不能再分裂
        for i in range(4):
            # 第i个属性的值检测，是否在一个划分区间内，是，则表示该属性不能再做划分条件
            column = array[:, i]
            min_v,max_v = column.min(),column.max()
            points = self.dp[i]
            for p in points:
                if min_v<p and max_v>p:
                    # 存在这样一个属性的划分点可以进行划分，则缓存该属性标号
                    result.append(i)
                    break

        # 类标号判定
        classes = list(array[:, 4])
        count_of_class = [classes.count(s) for s in set(classes)]
        # 这里本来是设置划分阈值0.95
        flag_class_can_divide = True if max(count_of_class)/len(classes)<0.95 else False

        # 仅当存在可划分的属性且类标号不同(0.9阈值)时，返回可作为划分的属性的列表，表示可划分
        if len(result)>0 and flag_class_can_divide:
            return result
        else:
            return False


    def insert(self,node):
        """插入子结点"""
        self.children.append(node)


    def divide(self):
        """
        就是根据属性测试条件，以及测试点，将数据集合分成几个部分，分别建立TreeNode然后加入子节点集合
        再递归调用子节点的划分函数，完成整个决策树的建立
        """
        if self.nodetype: # 若等于1(或2,空叶节点)，true，叶节点，不可划分
            return None

        # 根据属性划分
        print('-------------- divide -----------------')
        dpoints = self.dp[self.divideAttr]
        print('info: attr & divide points - ', self.divideAttr, dpoints)

        # 各划分区间的数据集
        recordset = []
        for i in range(len(dpoints) + 1):
            # 划分区间比划分点大1,为每个区间都创建一个子节点数据集
            # 注：此处，可能数据集为空，但是只表示训练集没有该数据，测试集可能会存在，故必须为这个空集创建一个叶结点，类标号为父节点的标号
            temp = []
            recordset.append(temp)

        # 遍历每个记录的该属性值，进行划分
        for record in self.data:
            # 当前记录所在的划分区间号
            gap = 0
            for p in dpoints:
                if record[self.divideAttr] > p:
                    gap += 1
            recordset[gap].append(record)

        # 为每个划分区间对应的数据集创建子节点
        for gap,noderecords in enumerate(recordset):
            child = None
            if len(noderecords)==0:
                # 空集合，创建type=2的空叶节点（特殊的叶节点）
                child = TreeNode([],gap,2,self.classlabel)
                self.insert(child)
            else:
                child = TreeNode(noderecords,gap)
                self.insert(child)

        # 递归调用子节点的划分函数
        for child in self.children:
            child.divide()


    def predict(self,data):
        """
        对一条数据做类型判定
        :param data: 输入一个离散化后的列表类型数据（包含4个属性）
        :return: 返回该数据的类型,(由于本案例中测试数据类标号已知，故同时作为模型检测函数，返回检测结果，True表示正确)
        """
        if self.nodetype==1 or self.nodetype==2:
            return self.classlabel,True if self.classlabel==data[4] else False
        else:
            # 根据划分属性，即测试数据该属性值的取值范围决定流向哪一个子节点
            rgap = 0
            for p in self.dp[self.divideAttr]:
                if data[self.divideAttr] > p:
                    rgap += 1

            nextnode = None
            for child in self.children:
                if child.gap == rgap:
                    # 找到了本结点
                    nextnode = child
                    break

            if nextnode:
                return nextnode.predict(data)
            else:
                print('没有找到对应的子节点...该数据测试出错')
                return 0,False


if __name__ == '__main__':
    trainingset, testset, dataset = dh.loadData()
    s_trainset = dh.discretization(trainingset)
    s_testset = dh.discretization(testset)
    s_dataset = dh.discretization(dataset)

    root = TreeNode(s_trainset)
    root.divide()
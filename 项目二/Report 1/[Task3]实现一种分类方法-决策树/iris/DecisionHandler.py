# 决策处理模块
import DataHandler as dh
from TreeNode import TreeNode
import numpy as np

def createDecisionTree(dataset:list):
    """
    决策树构造函数
    :param dataset:输入训练数据集
    :return: 返回当前决策树的root结点
    """
    # 创建树根节点
    tn = TreeNode(dataset)
    # 树根据数据集开始分裂
    tn.divide()
    # 返回最后分裂完成的决策树
    return tn

def TestDecisonTree(dataset:list,DTree:TreeNode):
    """
    决策树测试函数
    :param dataset:输入测试集
    :return: 返回该模型针对这个测试集的一系列性能测试结果的列表
    """
    result = []
    for data in dataset:
        result.append(predict(data,DTree))

    # 准确率
    npresult = np.array(result)
    r = list(npresult[:,1])
    print(npresult)
    return np.array([r.count(s) for s in set(r)]).max()/len(r)

def predict(data,DTree:TreeNode):
    """
    对一条数据做类型判定
    :param data: 输入一个离散化后的列表类型数据（包含4个属性）
    :return: 返回该数据的类型,(由于本案例中测试数据类标号已知，故同时作为模型检测函数，返回检测结果，True表示正确)
    """
    return DTree.predict(data)


if __name__ == '__main__':
    # 获取原始数据集
    trainingset, testset, dataset = dh.loadData()
    # 离散化处理,得到标准的处理数据standard data set
    s_trainset = dh.discretization(trainingset)
    s_testset = dh.discretization(testset)
    s_dataset = dh.discretization(dataset)
    print(s_dataset)

    # 用训练集构造决策树
    root = createDecisionTree(s_trainset)
    # 测试该决策树,怎么跟踪测试记录的流动，可见叶结点的分裂的属性测试条件应该被记录
    print('准确率：',TestDecisonTree(s_testset,root))
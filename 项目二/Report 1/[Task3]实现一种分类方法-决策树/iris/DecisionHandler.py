# 决策处理模块
import DataHandler as dh
from TreeNode import TreeNode

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

def testDecisonTree(dataset:list):
    """
    决策树测试函数
    :param dataset:输入测试集
    :return: 返回该模型针对这个测试集的一系列性能测试结果的列表
    """
    pass


def predict(data):
    """
    对一条数据做类型判定
    :param data: 输入一个列表类型数据（包含4个属性）
    :return: 返回该数据的类型
    """
    pass
    return 'type a'


if __name__ == '__main__':
    # 获取原始数据集
    trainingset, testset, dataset = dh.loadData()
    # 离散化处理,得到标准的处理数据standard data set
    s_trainset = dh.discretization(trainingset)
    s_testset = dh.discretization(testset)
    s_dataset = dh.discretization(dataset)


    # 用训练集构造决策树
    root = createDecisionTree(s_trainset)
    # 测试该决策树
    # TODO,怎么跟踪测试记录的流动，可见叶结点的分裂的属性测试条件应该被记录
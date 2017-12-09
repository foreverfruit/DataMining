'''
分类模块，主要完成以下几个功能：
1.剔除测试集中的属性不完整的对象
2.测试集的变换：按照训练集的映射方法对测试数据做变换
3.KNN算法：对测试集做预测
4.结果比对，分析错误率
'''
import numpy as np
from sklearn import neighbors
import  DataHandler as pp

def handle_normal(filename:str):

    cvt1 = lambda s: pp.dic_workclass.get(str(s, 'utf-8'))
    cvt3 = lambda s: pp.dic_education.get(str(s, 'utf-8'))
    cvt5 = lambda s: pp.dic_marital_status.get(str(s, 'utf-8'))
    cvt6 = lambda s: pp.dic_occupation.get(str(s, 'utf-8'))
    cvt7 = lambda s: pp.dic_relationship.get(str(s, 'utf-8'))
    cvt8 = lambda s: pp.dic_race.get(str(s, 'utf-8'))
    cvt9 = lambda s: pp.dic_sex.get(str(s, 'utf-8'))
    cvt13 = lambda s: pp.dic_native_country.get(str(s, 'utf-8'))
    cvt14 = lambda s: pp.dic_class.get(str(s, 'utf-8'))

    data = np.loadtxt(fname=filename, delimiter=', ', usecols=range(15), dtype=float,
                      converters={1: cvt1, 3: cvt3, 5: cvt5, 6: cvt6, 7: cvt7, 8: cvt8, 9: cvt9, 13: cvt13, 14: cvt14})

    return data

def handle_continuous(data:np.ndarray,filename:str):
    for i in range(data.shape[0]):
        # 1. age-0、hours_pw-12，采用5的区间映射到0到20的取值范围
        data[i,0] = data[i,0]/5
        data[i, 12] = data[i, 12] / 5

        # 2.fnlwgt-2,先取log运算，log后的取值范围[9,15]，再按0.5长度做等区间长度划分
        data[i,2] = (np.log(data[i,2])-9)/0.5

        # 3. cpl_gain-10、cpl_loss-11,简单做标记，对不为0的元素取1做标志
        data[i,10] = 1 if data[i,10]>0 else 0
        data[i, 11] = 1 if data[i, 11] > 0 else 0

    # 保存文件,精度取到小数点后8位
    np.savetxt(filename, data, fmt='%d', delimiter=',')

    return data

def do_classify(train_filename:str,test_filename:str):
    '''
    根据数据，做KNN预测，给出预测正确率
    '''
    # 训练集
    a = np.loadtxt(fname=train_filename+'.done', delimiter=',', dtype=float, unpack=True)
    train_attrs = np.column_stack((a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13]))
    train_class = a[14]
    # 训练
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_attrs, train_class)
    #knn.fit(a[:,0:14], a[:,14])

    # 测试集
    b = np.loadtxt(fname=test_filename+'.done', delimiter=',', dtype=float, unpack=True)
    test_attrs = np.column_stack((b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13]))
    test_class = b[14]
    # 预测
    result = knn.predict(test_attrs)
    #result = knn.predict(b[:,0:14])

    # 计算正确率
    count = 0
    for i in range(result.shape[0]):
        if test_class[i]==result[i]:
        #if b[i,14]==result[i]:
            count+=1

    print('result:',count/result.size)


if __name__ == '__main__':
    filename = 'adult.test'
    pp.discard_records(filename)
    data = handle_normal(filename+'.done')
    data = handle_continuous(data, filename+'.done')
    do_classify('adult.data',filename)

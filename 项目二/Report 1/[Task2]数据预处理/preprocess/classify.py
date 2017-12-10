'''
分类模块，主要完成以下几个功能：
1.剔除测试集中的属性不完整的对象
2.测试集的变换：按照训练集的映射方法对测试数据做变换
3.KNN算法：对测试集做预测
4.结果比对，分析错误率
'''
import numpy as np
from sklearn import neighbors
import  preprocess as pp
import math

# 标称值映射关系
map_workclass = {1.0: 0.21879206676837476, 2.0: 0.2857142857142857, 3.0: 0.55865921787709494, 4.0: 0.32358803986710966, 5.0: 0.26896012509773259, 6.0: 0.0}
map_education = {1.0: 0.42149088025376685, 2.0: 0.20005989817310571, 3.0: 0.056297709923664119, 4.0: 0.16432926829268293, 5.0: 0.74907749077490771, 6.0: 0.25396825396825395, 7.0: 0.26319816373374139, 8.0: 0.054945054945054944, 9.0: 0.06283662477558348, 10.0: 0.076923076923076927, 11.0: 0.56422864167178854, 12.0: 0.039735099337748346, 13.0: 0.071951219512195116, 14.0: 0.7466666666666667, 15.0: 0.036036036036036036}
map_marital = {1.0: 0.45495911837895486, 2.0: 0.10726150925486473, 3.0: 0.048324079786140242, 4.0: 0.070287539936102233, 5.0: 0.096735187424425634, 6.0: 0.083783783783783788, 7.0: 0.47619047619047616}
map_occupation ={1.0: 0.30482456140350878, 2.0: 0.22531017369727047, 3.0: 0.041095890410958902, 4.0: 0.27064732142857145, 5.0: 0.48522044088176353, 6.0: 0.44848935116394256, 7.0: 0.061481481481481484, 8.0: 0.12461851475076297, 9.0: 0.13383499059392637, 10.0: 0.11627906976744186, 11.0: 0.20292620865139949, 12.0: 0.006993006993006993, 13.0: 0.32608695652173914, 14.0: 0.1111111111111111}
map_relationship = {1.0: 0.49359886201991465, 2.0: 0.014330497089117778, 3.0: 0.45566877958757923, 4.0: 0.10652342738804038, 5.0: 0.03937007874015748, 6.0: 0.066313823163138233}
map_race = {1.0: 0.26371804264836307, 2.0: 0.27709497206703909, 3.0: 0.11888111888111888, 4.0: 0.090909090909090912, 5.0: 0.12992545260915869}
map_sex = {1.0: 0.11367818442036394, 2.0: 0.3138370951913641}
map_country = {1.0: 0.25432664339732403, 2.0: 0.3888888888888889, 3.0: 0.34883720930232559, 4.0: 0.11009174311926606, 5.0: 0.3364485981308411, 6.0: 0.34375, 7.0: 0.0, 8.0: 0.40000000000000002, 9.0: 0.38983050847457629, 10.0: 0.27586206896551724, 11.0: 0.19718309859154928, 12.0: 0.35454545454545455, 13.0: 0.27173913043478259, 14.0: 0.42857142857142855, 15.0: 0.083333333333333329, 16.0: 0.31914893617021278, 17.0: 0.35294117647058826, 18.0: 0.19642857142857142, 19.0: 0.125, 20.0: 0.078125, 21.0: 0.054098360655737705, 22.0: 0.11764705882352941, 23.0: 0.20833333333333334, 24.0: 0.44444444444444442, 25.0: 0.029850746268656716, 26.0: 0.11764705882352941, 27.0: 0.14814814814814814, 28.0: 0.18181818181818182, 29.0: 0.095238095238095233, 30.0: 0.035714285714285712, 31.0: 0.23076923076923078, 32.0: 0.047619047619047616, 33.0: 0.060606060606060608, 34.0: 0.17647058823529413, 35.0: 0.375, 36.0: 0.089999999999999997, 37.0: 0.1111111111111111, 38.0: 0.066666666666666666, 39.0: 0.31578947368421051, 40.0: 0.0}


def handle_normal(filename:str):
    filename = filename+'.done'

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
    # 遍历行向量，并根据映射表做变换
    for row,line in enumerate(data):
        data[row,1] = map_workclass.get(data[row,1])
        data[row,3] = map_education.get(data[row,3])
        data[row,5] = map_marital.get(data[row,5])
        data[row,6] = map_occupation.get(data[row,6])
        data[row,7] = map_relationship.get(data[row,7])
        data[row,8] = map_race.get(data[row,8])
        data[row,9] = map_sex.get(data[row,9])
        data[row,13] = map_country.get(data[row,13])

    return data

def handle_continuous(data:np.ndarray,filename:str):
    for i in range(data.shape[0]):
        # age，年龄（17,90），直接除以100
        data[i,0] = data[i,0]/100
        # hours_pw，（1,99），直接除以100
        data[i, 12] = data[i, 12]/100
        # edu_num，（1,16），
        data[i, 4] = (data[i, 4]-1) / 15
        # fnlwgt-2,log后的值域为(9.5,14.2),取整到（9,15）
        data[i,2] = (np.log(data[i,2])-9)/6
        # cpl_loss/gain,(0,4356)
        data[i, 10] = data[i, 11] / (data[i, 10] + 1)
        data[i, 11] = -1
        data[i,10] = data[i,10]/4356
        # 越界处理,越界的直接取0或1
        for index,v in enumerate(data[i]):
            if index in [0,2,4,10,12]:
                if data[i,index]<0:
                    data[i,index]=0
                if data[i,index]>1:
                    data[i,index]=1

    # 保存文件,精度取到小数点后8位
    np.savetxt(filename+'.done', data, fmt='%.8f', delimiter=',')
    return data

def do_classify(train_filename:str,test_filename:str):
    '''
    根据数据，做KNN预测，给出预测正确率
    '''
    # 训练集
    a = np.loadtxt(fname=train_filename+'.done', delimiter=',', dtype=float, unpack=True)
    train_attrs = np.column_stack((a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[12],a[13]))
    train_class = a[14]

    # 训练
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_attrs, train_class)

    # 测试集
    b = np.loadtxt(fname=test_filename+'.done', delimiter=',', dtype=float, unpack=True)
    test_attrs = np.column_stack((b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[12],b[13]))
    test_class = b[14]
    # 预测
    result = knn.predict(test_attrs)

    # 计算正确率
    count = 0
    for i in range(result.shape[0]):
        if test_class[i]==result[i]:
            count+=1

    print('result:',count/result.size)


def my_classify1(train_filename:str,test_filename:str):
    '''
    我的分类算法：懒惰法，粗糙模型，暴力比较距离
    结果：错误，没有平衡不同类数据量对力的影响，如A类10000个，B类10个，那么K取多少，才能决定了测试数据的位置和最终预测的分类的关系
    '''
    G = 1
    K = 2

    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    train_a = train[np.where(train[:, 14] == 0)]
    train_b = train[np.where(train[:, 14] == 1)]
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('a类占比:',size_a/train.shape[0])


    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)

    cors = []
    # 计算训练集每个维度相关度
    for i in range(train.shape[1]-1):
        if i==11:
            # 跳过列11，但是为了后面的计算，这个值还是要占个位，否则后续的计算会数组越界
            cors.append(-1)
            continue
        cors.append(np.corrcoef(train[:,i],train[:,14])[0,1])

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    for i in test:
        # 当前测试对象额实际类标号
        class_test = i[14]
        # 当前测试对象的预测标号，大于0表示a类，小于0表示b类
        var_predict = 0
        # 遍历每一个属性（维度）
        for j in range(test.shape[1]-1):
            # 忽略列为11的维度
            if j==11:
                continue
            var_test = i[j]
            # 该属性维度合力为F=0
            F_a = 0
            # 计算训练集中类A的合力
            for a in train_a:
                var_train = a[j]
                f = (G-abs(var_train-var_test)*G)/(1+abs(var_train-var_test)**K)
                F_a+=f
            # 计算训练集中类B的合力
            F_b = 0
            for b in train_b:
                var_train = b[j]
                f = (G-abs(var_train-var_test)*G)/(1+abs(var_train-var_test)**K)
                F_b+=f

            # 合力比较，乘以权重
            var_predict+=(F_a/size_a-F_b/size_b)*abs(cors[j])

        # 预测的类标号
        predict_class = 0
        # 计量结果
        if var_predict == 0:
            # 预测失败
            print('failed!')
            fail_count += 1
        elif var_predict>0:
            # 预测为a类，标号为0
            predict_class = 0
        elif var_predict<0:
            predict_class = 1

        print('结果：','正确' if predict_class==class_test else '错误','预测类：',predict_class,'预测值：',var_predict,'真实类：',class_test)
        if(predict_class == class_test):
            count += 1

    size = test.shape[0]
    print('测试总数：',size,'失败率：',fail_count/size,'准确率:',count/size)


def my_classify2(train_filename:str,test_filename:str):
    '''
    我的分类算法：距离法，高维空间中点的平均距离做比较，距离越小，表示越吻合
    准确率：100%
    '''
    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    train_a = train[np.where(train[:, 14] == 0)]
    train_b = train[np.where(train[:, 14] == 1)]
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('a类占比:',size_a/train.shape[0])

    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    size_test = test.shape[0]
    for index in range(size_test):
        element_test = test[index]
        # 当前测试对象额实际类标号
        class_test = element_test[14]
        # 到两个类的平均距离
        d_a,d_b = 0,0
        # 遍历到两个类的训练样本的平均距离
        for element_train in train_a:
            minus = element_test-element_train
            minus = np.square(minus)
            d_a += np.sqrt(minus.sum())

        for element_train in train_b:
            minus = element_test-element_train
            minus = np.square(minus)
            d_b += np.sqrt(minus.sum())

        # 预测的类标号
        predict_class = 0
        # 计量结果
        if d_a/size_a == d_b/size_b:
            # 预测失败
            print('failed!')
            fail_count += 1
        elif d_a/size_a < d_b/size_b:
            # 预测为a类，标号为0
            predict_class = 0
        elif d_a/size_a > d_b/size_b:
            predict_class = 1

        print('结果：','正确' if predict_class==class_test else '错误','预测类：',predict_class,'真实类：',class_test)
        if(predict_class == class_test):
            count += 1

    #size_test = test.shape[0]
    print('测试总数：',size_test,'失败率：',fail_count/size_test,'准确率:',count/size_test)


def my_classify3(train_filename:str,test_filename:str):
    '''
    我的分类算法：距离法，高维空间中点的平均距离做比较，距离越小，表示越吻合
    方法二的基础上，提速，方法：合成向量（各维累加）
    1.各维相加，准确率：0.533，显然太简单，物理解释也能说明这种方法的不合理，距离会随着各维值的累加而急剧增加，最终严重受样本容量的影响
    2.各维取平均，classify4
    '''
    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    train_a = train[np.where(train[:, 14] == 0)]
    train_b = train[np.where(train[:, 14] == 1)]

    # 类统计
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('a类占比:',size_a/train.shape[0])

    # 各类的向量和
    train_a_sum = np.sum(train_a,axis=0)
    train_b_sum = np.sum(train_b,axis=0)

    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    size_test = test.shape[0]
    for index in range(size_test):
        element_test = test[index]
        # 当前测试对象额实际类标号
        class_test = element_test[14]
        # 到两个类的平均距离
        d_a,d_b = 0,0
        # 遍历到两个类的训练样本的平均距离
        element_a = element_test*size_a
        element_b = element_test*size_b
        d_a = np.sqrt(np.square(element_a-train_a_sum).sum())
        d_b = np.sqrt(np.square(element_b-train_b_sum).sum())

        # 预测的类标号
        predict_class = 0
        # 计量结果
        if d_a == d_b:
            # 预测失败
            print('failed!')
            fail_count += 1
        elif d_a < d_b:
            # 预测为a类，标号为0
            predict_class = 0
        elif d_a > d_b:
            predict_class = 1

        print('结果：','正确' if predict_class==class_test else '错误','预测类：',predict_class,'真实类：',class_test)
        if(predict_class == class_test):
            count += 1

    #size_test = test.shape[0]
    print('测试总数：',size_test,'失败率：',fail_count/size_test,'准确率:',count/size_test)


def my_classify4(train_filename:str,test_filename:str):
    '''
    我的分类算法：距离法，高维空间中点的平均距离做比较，距离越小，表示越吻合
    方法3的改进
    '''
    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    train_a = train[np.where(train[:, 14] == 0)]
    train_b = train[np.where(train[:, 14] == 1)]

    # 类统计
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('a类占比:',size_a/train.shape[0])

    # 各类的平均向量
    train_a_avg = np.sum(train_a,axis=0)/size_a
    train_b_avg = np.sum(train_b,axis=0)/size_b

    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    size_test = test.shape[0]
    for index in range(size_test):
        element_test = test[index]
        # 当前测试对象额实际类标号
        class_test = element_test[14]
        # 到两个类的平均距离
        d_a,d_b = 0,0
        d_a = np.sqrt(np.square(element_test-train_a_avg).sum())
        d_b = np.sqrt(np.square(element_test-train_b_avg).sum())

        # 预测的类标号
        predict_class = 0
        # 计量结果
        if d_a == d_b:
            # 预测失败
            print('failed!')
            fail_count += 1
        elif d_a < d_b:
            # 预测为a类，标号为0
            predict_class = 0
        elif d_a > d_b:
            predict_class = 1

        print('结果：','正确' if predict_class==class_test else '错误','预测类：',predict_class,'真实类：',class_test)
        if(predict_class == class_test):
            count += 1

    #size_test = test.shape[0]
    print('测试总数：',size_test,'失败率：',fail_count/size_test,'准确率:',count/size_test)



def my_classify5(train_filename:str,test_filename:str):
    '''
    我的分类算法：距离法
    各维度平均距离，根据相关度做加权
    '''
    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    train_a = train[np.where(train[:, 14] == 0)]
    train_b = train[np.where(train[:, 14] == 1)]

    # 类统计
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('a类占比:',size_a/train.shape[0])

    # 各类的平均向量
    train_a_avg = np.sum(train_a,axis=0)/size_a
    train_b_avg = np.sum(train_b,axis=0)/size_b

    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)

    cors = []
    # 计算训练集每个维度相关度
    for i in range(train.shape[1] - 1):
        if i == 11:
            # 跳过列11，但是为了后面的计算，这个值还是要占个位，否则后续的计算会数组越界
            cors.append(0)
            continue
        cors.append(np.corrcoef(train[:, i], train[:, 14])[0, 1])

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    size_test = test.shape[0]
    for index in range(size_test):
        element_test = test[index]
        # 当前测试对象实际类标号
        class_test = element_test[14]
        # 当前测试对象的预测标号，大于0表示a类，小于0表示b类
        var_predict = 0
        # 遍历每一个属性（维度）
        for j in range(test.shape[1] - 1):
            # 忽略列为11的维度
            if j == 11:
                continue

            # 距离比较，乘以权重
            var_predict += (abs(element_test[j]-train_a_avg[j]) - abs(element_test[j]-train_b_avg[j])) * abs(cors[j])

        # 预测的类标号
        predict_class = 0
        # 计量结果
        if var_predict == 0:
            # 预测失败
            print('failed!')
            fail_count += 1
        elif var_predict > 0:
            # 预测为a类，标号为0,计算是a的距离减去b的距离做加权，大于0，表示离a比较远，更倾向与b，b的类标号为1
            predict_class = 1
        elif var_predict < 0:
            predict_class = 0

        print('结果：', '正确' if predict_class == class_test else '错误', '预测类：', predict_class, '预测值：', var_predict, '真实类：',
              class_test)
        if (predict_class == class_test):
            count += 1

    #size_test = test.shape[0]
    print('测试总数：',size_test,'失败率：',fail_count/size_test,'准确率:',count/size_test)


if __name__ == '__main__':
    filename = 'adult.test'
    pp.discard(filename)
    data = handle_normal(filename)
    data = handle_continuous(data, filename)
    #do_classify('adult.data',filename)      # 81%，scikit-learn的KNN调用
    #my_classify1('adult.data',filename)     # 76%，错误的模型……
    #my_classify2('adult.data',filename)     # 100%，速度巨慢，每一个测试实体计算对所有训练实体的距离再求平均
    #my_classify3('adult.data',filename)     # 53%，训练向量求和，将测试向量相应扩大size倍，再求一次距离，距离剧烈扩大，结果显然不正确
    #my_classify4('adult.data',filename)     # 100% 且速度最快，秒出结果，先求训练总体的平均向量，再求一次距离即可
    my_classify5('adult.data',filename)      # 72%，各维距离=测试样本各维度值-各维度训练样本均值，再根据各维度相关度加权，最后属于距离较小的类
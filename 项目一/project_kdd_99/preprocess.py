'''
预处理模块
41个属性+1个类标号,详情见feature list
其中，类标号，正常为normal，攻击：训练集22个，还有17个只出现在测试集
'''

import numpy as np
from sklearn import neighbors
import os

'''
# 精细的类标号，弃用
dict_class_labels = {'normal':0,'back':1,'buffer_overflow':2,'ftp_write':3,
                    'guess_passwd':4,'imap':5,'ipsweep':6,'land':7,'loadmodule':8,
                    'multihop':9,'neptune':10,'nmap':11,'perl':12,'phf':13,'pod':14,
                    'portsweep':15,'rootkit':16,'satan':17,'smurf':18,'spy':19,
                    'teardrop':20,'warezclient':21,'warezmaster':22,'others':23}
'''
# 以下dict根据get_str_dict()统计而来
dict_protocal = {'icmp': 0.0045415758704099411, 'tcp': 0.40414068871175651, 'udp': 0.94217352854475778}
dict_service = {'IRC': 0.97674418604651159, 'X11': 0.81818181818181823, 'Z39_50': 0.0, 'auth': 0.67073170731707321, 'bgp': 0.0, 'courier': 0.0, 'csnet_ns': 0.0, 'ctf': 0.0, 'daytime': 0.0, 'discard': 0.0, 'domain': 0.025862068965517241, 'domain_u': 0.99982943885382913, 'echo': 0.0, 'eco_i': 0.23690621193666261, 'ecr_i': 0.0012260127931769723, 'efs': 0.0, 'exec': 0.0, 'finger': 0.69850746268656716, 'ftp': 0.46741854636591479, 'ftp_data': 0.80449057403092561, 'gopher': 0.0, 'hostnames': 0.0, 'http': 0.96256202074875963, 'http_443': 0.0, 'imap4': 0.0, 'iso_tsap': 0.0, 'klogin': 0.0, 'kshell': 0.0, 'ldap': 0.0, 'link': 0.0, 'login': 0.0, 'mtp': 0.0, 'name': 0.0, 'netbios_dgm': 0.0, 'netbios_ns': 0.0, 'netbios_ssn': 0.0, 'netstat': 0.0, 'nnsp': 0.0, 'nntp': 0.0, 'ntp_u': 1.0, 'other': 0.77822302058864168, 'pm_dump': 0.0, 'pop_2': 0.0, 'pop_3': 0.3910891089108911, 'printer': 0.0, 'private': 0.066424391079689435, 'red_i': 1.0, 'remote_job': 0.0, 'rje': 0.0, 'shell': 0.0089285714285714281, 'smtp': 0.98714388563200661, 'sql_net': 0.0, 'ssh': 0.0095238095238095247, 'sunrpc': 0.0, 'supdup': 0.0, 'systat': 0.0, 'telnet': 0.42690058479532161, 'tftp_u': 1.0, 'tim_i': 0.2857142857142857, 'time': 0.33121019108280253, 'urh_i': 1.0, 'urp_i': 0.9981412639405205, 'uucp': 0.0, 'uucp_path': 0.0, 'vmnet': 0.0, 'whois': 0.0}
dict_flag  = {'OTH': 0.125, 'REJ': 0.19873488372093023, 'RSTO': 0.1157167530224525, 'RSTOS0': 0.0, 'RSTR': 0.034330011074197121, 'S0': 0.00058615973427425376, 'S1': 0.94736842105263153, 'S2': 0.70833333333333337, 'S3': 0.69999999999999996, 'SF': 0.24233431983934045, 'SH': 0.0}
dict_class_label = {'normal.':0,'others':1}

# preprocess_prior统计出来的各属性min和max，用于归一化
attr_arrange = {0: [0.0, 58329.0], 1: [0.0045415758704099411, 0.94217352854475778], 2: [0.0, 1.0], 3: [0.0, 0.94736842105263153], 4: [0.0, 693375640.0], 5: [0.0, 5155468.0], 6: [0.0, 1.0], 7: [0.0, 3.0], 8: [0.0, 3.0], 9: [0.0, 30.0], 10: [0.0, 5.0], 11: [0.0, 1.0], 12: [0.0, 884.0], 13: [0.0, 1.0], 14: [0.0, 2.0], 15: [0.0, 993.0], 16: [0.0, 28.0], 17: [0.0, 2.0], 18: [0.0, 8.0], 19: [0.0, 0.0], 20: [0.0, 0.0], 21: [0.0, 1.0], 22: [0.0, 511.0], 23: [0.0, 511.0], 24: [0.0, 1.0], 25: [0.0, 1.0], 26: [0.0, 1.0], 27: [0.0, 1.0], 28: [0.0, 1.0], 29: [0.0, 1.0], 30: [0.0, 1.0], 31: [0.0, 255.0], 32: [0.0, 255.0], 33: [0.0, 1.0], 34: [0.0, 1.0], 35: [0.0, 1.0], 36: [0.0, 1.0], 37: [0.0, 1.0], 38: [0.0, 1.0], 39: [0.0, 1.0], 40: [0.0, 1.0]}


def get_str_dict():
    '''
    统计symbol属性根据其相关度（分类中的概率）得出映射值，形成字典用于将数据从str映射到归一化后的float
    统计是根据原数据集trainset_full做的
    **trainset_full太大，内存溢出了，暂时统计是根据10percent数据做的**
    '''
    #filename = 'trainset_full'
    #dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocess.py')))
    #filename = dir + os.path.sep + filename

    filename = 'trainset_10_percent'
    data = np.loadtxt(filename, dtype=str, delimiter=',', usecols=(1, 2, 3,41), unpack=False)
    # 所有类标号为'normal.'的行向量组成一个新的矩阵用于下文计算频率
    data_class_normal = data[data[:, 3] == 'normal.', :]
    attr_names = ['protocal', 'service', 'flag']

    print('----------映射关系：------------')
    # 遍历所有属性
    for i in range(len(attr_names)):
        # 只取标称属性做处理
        attr_name = attr_names[i]
        # 选取该属性维度（列）
        column = data[:, i]
        column_class1 = data_class_normal[:, i]
        # unique返回该属性的value，及各value对应的counts
        values, counts = np.unique(column, return_counts=True)
        # 映射表，key为属性值value，value为该属性值对应的频率freq，最终用freq替代matrix中的value
        mapping = {}
        # 遍历计算每一个value的类标号为1的频率
        for index, v in enumerate(values):
            # 分别计算：value，value对应的count，以及类标号为1的value的count
            value = values[index]
            count_all = counts[index]
            count_class1 = column_class1.tolist().count(value)
            freq = count_class1 / count_all
            # 添加该value的映射
            mapping[value] = freq

        print('dict_' + attr_name, mapping)


def preprocess_prior(filename:str): # 预处理前的统计函数
    cvt1 = lambda s: dict_protocal.get(str(s, 'utf-8'),0)
    cvt2 = lambda s: dict_service.get(str(s, 'utf-8'),0)
    cvt3 = lambda s: dict_flag.get(str(s, 'utf-8'),0)
    cvt41 = lambda s: dict_class_label.get(str(s,'utf-8'),1)
    data = np.loadtxt(fname=filename, delimiter=',', dtype=float,converters={1: cvt1, 2: cvt2, 3: cvt3, 41: cvt41})
    min_max = {}
    # 求各维度最小值、最大值
    for i in range(41):
        dim = data[:,i]
        min = dim.min()
        max = dim.max()
        min_max[i] = [min,max]
        #temp = (max-min) if max!=min else 1
        #data[:,i] = (data[:,i]-min)/temp
        # print('第'+str(i)+'个属性：','before：',min,max,'after:',data[:,i].min(),data[:,i].max())

    #np.clip(data, 0, 1)
    #print(min_max)


def preprocess(filename:str):
    '''
    预处理：
    1.加载数据，将str数据根据之前统计出来的映射表做映射
    2.其他非str值的symbol属性先不管
    3.对其他所有属性（除了1，2,3,41）做归一化处理
    '''
    # step 1
    cvt1 = lambda s: dict_protocal.get(str(s, 'utf-8'),0)
    cvt2 = lambda s: dict_service.get(str(s, 'utf-8'),0)
    cvt3 = lambda s: dict_flag.get(str(s, 'utf-8'),0)
    cvt41 = lambda s: dict_class_label.get(str(s,'utf-8'),1)

    data = np.loadtxt(fname=filename, delimiter=',', dtype=float,converters={1: cvt1, 2: cvt2, 3: cvt3, 41: cvt41})

    # step 3
    # 求各维度最小值、最大值
    for i in range(41):
        # 这个temp为了处理分母为0的情况
        min_max = attr_arrange.get(i)
        temp = (min_max[1]-min_max[0]) if min_max[1]!=min_max[0] else 1
        data[:,i] = (data[:,i]-min_max[0])/temp

    # 越界处理，使元素都在[0,1]内
    np.clip(data, 0, 1)
    # 保存归一化后的数据文件
    np.savetxt(filename + '.done', data, fmt='%.5f', delimiter=',')


def classify_KNN(train_filename,test_filename):
    '''
    根据数据，做KNN预测，给出预测正确率
    '''
    # 训练集
    a = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=True)
    train_attrs = np.column_stack((a[range(41)]))
    train_class = a[41]

    # 训练
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_attrs, train_class)

    # 测试集
    b = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=True)
    test_attrs = np.column_stack((b[range(41)]))
    test_class = b[41]
    # 预测
    result = knn.predict(test_attrs)

    # 计算正确率
    count = (result-test_class).tolist().count(0)

    print('result:', count / result.size,count)


def my_classify(train_filename:str,test_filename:str):
    '''
    我的分类算法：距离法，高维空间中点的平均距离做比较，距离越小，表示越吻合
    '''
    # 训练集
    train = np.loadtxt(fname=train_filename + '.done', delimiter=',', dtype=float, unpack=False)
    # normal,正常连接类
    train_a = train[np.where(train[:, 41] == 0)]
    # 攻击连接类
    train_b = train[np.where(train[:, 41] == 1)]

    # 类统计
    size_a,size_b = train_a.shape[0],train_b.shape[0]
    print('训练集a类占比:',size_a/train.shape[0])

    # 各类的平均向量
    train_a_avg = np.sum(train_a,axis=0)/size_a
    train_b_avg = np.sum(train_b,axis=0)/size_b

    # 测试集
    test = np.loadtxt(fname=test_filename + '.done', delimiter=',', dtype=float, unpack=False)
    test_a = train[np.where(train[:, 41] == 0)]
    test_b = train[np.where(train[:, 41] == 1)]
    size_test_a, size_test_b = test_a.shape[0], test_b.shape[0]
    print('测试集a类占比:', size_test_a / test.shape[0])

    count = 0
    fail_count = 0
    # 遍历测试每一个测试数据
    size_test = test.shape[0]
    for index in range(size_test):
        element_test = test[index]
        # 当前测试对象额实际类标号
        class_test = element_test[41]
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

        #print('结果：','正确' if predict_class==class_test else '错误','预测类：',predict_class,'真实类：',class_test)
        if(predict_class == class_test):
            count += 1

    print('测试总数：',size_test,'失败率：',fail_count/size_test,'准确率:',count/size_test)


if __name__ == '__main__':

    train_filename = 'trainset_10_percent'
    test_filename = 'testset_with_labels'

    # 预处理时的中间函数：统计数据用的
    #get_str_dict()
    #preprocess_prior(train_filename)

    # 预处理函数
    #preprocess(train_filename)
    #preprocess(test_filename)

    # 分类函数1
    # scikit-learn的knn算法，在未归一化时，运行半小时，准确率92%
    #classify_KNN(train_filename,test_filename)

    # 分类函数2,运算时间很快，几秒
    # 训练集a类占比: 0.19691065764410826
    # 测试集a类占比: 0.31276183249793427
    # 测试总数： 311029 失败率： 0.0 准确率: 0.9214639149404075
    my_classify(train_filename,test_filename)
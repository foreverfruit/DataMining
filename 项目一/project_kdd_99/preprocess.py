'''
预处理模块
41个属性+1个类标号,详情见feature list
其中，类标号，正常为normal，攻击：训练集22个，还有17个只出现在测试集
'''

import numpy as np
import os

'''
# 精细的类标号，弃用
dict_class_labels = {'normal':0,'back':1,'buffer_overflow':2,'ftp_write':3,
                    'guess_passwd':4,'imap':5,'ipsweep':6,'land':7,'loadmodule':8,
                    'multihop':9,'neptune':10,'nmap':11,'perl':12,'phf':13,'pod':14,
                    'portsweep':15,'rootkit':16,'satan':17,'smurf':18,'spy':19,
                    'teardrop':20,'warezclient':21,'warezmaster':22,'others':23}
'''

def get_str_dict():
    '''
    统计symbol属性根据其相关度（分类中的概率）得出映射值，形成字典用于将数据从str映射到归一化后的float
    统计是根据原数据集trainset_full做的
    '''
    filename = 'trainset_full'
    dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocess.py')))
    filename = dir + os.path.sep + filename
    all_data = np.loadtxt(filename, dtype=str, delimiter=',', usecols=(1, 2, 3, 41), unpack=False)

    class_normal = all_data[np.where(all_data[:,3] == 'normal.')]

    attr_names = ['protocal','service','flag']
    for index,attr in enumerate(attr_names):
        # 计算class_normal中，各属性值的count,存于dict_class_normal中
        unique, count = np.unique(class_normal[:, index], return_counts=True)
        dict_class_normal = {}
        for i in range(unique.size):
            dict_class_normal[unique[i]] = count[i]

        # 计算总体中，各属性值的count,然后用对应的class_normal中的该值相除，得出比率，存于dict_result中
        unique, count = np.unique(all_data[:, index], return_counts=True)
        dict_result = {}
        for i in range(unique.size):
            dict_result[unique[i]] = dict_class_normal[unique[i]] / count[i]

        # 输出结果
        print('dict_'+attr,dict_result)




def handle_symbol(filename:str):
    '''
    symbol类型的数据处理:[column_num.attr_name:data_type]
    1.protocol_type:str
    2.service:str
    3.flag:str
    6.land
    11.logged_in
    20.is_host_login
    21.is_guest_login
    41.class_label:str
    symbol属性处理分为两步：1.str映射到int。2.根据相关度将int映射到具体的float取值（class_label不处理）
    '''
    np.loadtxt(filename,dtype=float,delimiter=',',converters={})


if __name__ == '__main__':
    get_str_dict()
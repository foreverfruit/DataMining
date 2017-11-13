"""
数据集格式：
0.age: continuous.
1.workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, 
            Local-gov, State-gov, Without-pay, Never-worked.
2.fnlwgt: continuous.
3.education: Bachelors, Some-college, 11th, HS-grad, Prof-school, 
            Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 
            10th, Doctorate, 5th-6th, Preschool.
4.education-num: continuous.
5.marital-status: Married-civ-spouse, Divorced, Never-married, Separated, 
           Widowed, Married-spouse-absent, Married-AF-spouse.
6.occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, 
            Prof-specialty, Handlers-cleaners, Machine-op-inspct, 
            Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, 
            Protective-serv, Armed-Forces.
7.relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
8.race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
9.sex: Female, Male.
10.capital-gain: continuous.
11.capital-loss: continuous.
12.hours-per-week: continuous.
13.native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, 
            Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, 
            Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, 
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, 
            Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, 
            Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, 
            Peru, Hong, Holand-Netherlands.
14.class
"""

# 数据预处理模块
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# 数据映射字典（字符串型数据到整形映射）
dic_workclass = {'Private':1, 'Self-emp-not-inc':2, 'Self-emp-inc':3, 'Federal-gov':4,
                 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7}
dic_education = {'Bachelors':1, 'Some-college':2, '11th':3, 'HS-grad':4, 'Prof-school':5,
                 'Assoc-acdm':6, 'Assoc-voc':7, '9th':8, '7th-8th':9, '12th':10, 'Masters':11,
                 '1st-4th':12, '10th':13, 'Doctorate':14, '5th-6th':15, 'Preschool':15}
dic_marital_status= {'Married-civ-spouse':1, 'Divorced':2, 'Never-married':3, 'Separated':4,
                'Widowed':5, 'Married-spouse-absent':6, 'Married-AF-spouse':7}
dic_occupation = {'Tech-support':1, 'Craft-repair':2, 'Other-service':3, 'Sales':4,
                  'Exec-managerial':5, 'Prof-specialty':6, 'Handlers-cleaners':7,
                  'Machine-op-inspct':8, 'Adm-clerical':9, 'Farming-fishing':10, 'Transport-moving':11,
                  'Priv-house-serv':12,  'Protective-serv':13, 'Armed-Forces':14}
dic_relationship = {'Wife':1, 'Own-child':2, 'Husband':3, 'Not-in-family':4,
                    'Other-relative':5, 'Unmarried':6}
dic_race = {'White':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4, 'Black':5}
dic_sex = {'Female':1, 'Male':2}
dic_native_country =  {'United-States':1, 'Cambodia':2, 'England':3, 'Puerto-Rico':4,
                       'Canada':5, 'Germany':6, 'Outlying-US(Guam-USVI-etc)':7, 'India':8,
                       'Japan':9, 'Greece':10, 'South':11, 'China':12, 'Cuba':13, 'Iran':14,
                       'Honduras':15, 'Philippines':16, 'Italy':17, 'Poland':18, 'Jamaica':19,
                       'Vietnam':20, 'Mexico':21, 'Portugal':22, 'Ireland':23, 'France':24,
                       'Dominican-Republic':25, 'Laos':26, 'Ecuador':27, 'Taiwan':12, 'Haiti':29,
                       'Columbia':30, 'Hungary':31, 'Guatemala':32, 'Nicaragua':33, 'Scotland':28,
                       'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37,
                       'Peru':38, 'Hong':39, 'Holand-Netherlands':40}
dic_class = {'<=50K':0,'>50K':1}

attr_names = ('age','workclass','fnlwgt','education','education-num','marital-status','occupation',
              'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class')

def remove_unknow_records(fname:str):
    """
    本函数仅完成数据文件分离工作，*.wiped用于下一步挖掘工作,*.dirty用于缺失记录的进一步统计
    文件操作：剔除含有未知属性的记录
    :param fname: 原始数据文件
    """
    # 需要操作的三个文件：原文件，清洗后的文件，脏数据（属性缺失）文件
    file = open(fname, 'r')
    f_wiped = open(fname+'.wiped','w')
    f_dirty = open(fname+'.dirty','w')

    # 缺失记录数量统计
    total,count = 0,0
    line = file.readline()
    while line:
        total+=1
        if line.__contains__('?'):
            count+=1
            f_dirty.write(line)
        else:
            f_wiped.write(line)
        line = file.readline()

    # 输出缺失记录的统计
    print('dirty records info:',count,count/total)
    f_dirty.write('缺失属性记录数：'+str(count)+'，占比：'+str(count/total))

    file.close()
    f_wiped.close()
    f_dirty.close()


def load_data(fname:str):
    """
    读取数据文件,完成字符串数据到int数据的映射
    :param name: 数据文件名
    :return: ndarray类型的多维数组
    """
    # 构造字符串到int的映射函数，用于numpy读取数据
    cvt1 = lambda s:dic_workclass.get(str(s,'utf-8'))
    cvt3 = lambda s: dic_education.get(str(s, 'utf-8'))
    cvt5 = lambda s: dic_marital_status.get(str(s, 'utf-8'))
    cvt6 = lambda s: dic_occupation.get(str(s, 'utf-8'))
    cvt7 = lambda s: dic_relationship.get(str(s, 'utf-8'))
    cvt8 = lambda s: dic_race.get(str(s, 'utf-8'))
    cvt9 = lambda s: dic_sex.get(str(s, 'utf-8'))
    cvt13 = lambda s: dic_native_country.get(str(s, 'utf-8'))
    cvt14 = lambda s: dic_class.get(str(s,'utf-8'))

    data =  np.loadtxt(fname=fname, delimiter=', ',usecols=range(15),dtype=int,
                       converters={1:cvt1,3:cvt3,5:cvt5,6:cvt6,7:cvt7,8:cvt8,9:cvt9,13:cvt13,14:cvt14})
    return data

def load_np_file(fname):
    """
    读取经过numpy整数化之后输出的数据文件
    :param fname: 文件名
    :return: 数据对应的ndarray类型多维数组
    """
    return np.loadtxt(fname,int,delimiter=',')


def statistics(data:ndarray):
    """
    对数据集做统计：最大值，最小值，平均值，方差，相关性，对于某些属性选择性做统计，如国家，只需统计相关性
    :param data:
    :return:
    """

    for index,datalist in enumerate(data):



if __name__ == '__main__':
    #remove_unknow_records('adult.data')
    data = load_data('adult.data.wiped')
    np.savetxt('preprocessed',data,fmt='%d',delimiter=',')

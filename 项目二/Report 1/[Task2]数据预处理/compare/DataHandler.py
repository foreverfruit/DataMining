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
from mpl_toolkits.mplot3d import Axes3D
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

def discard_records(fname:str):
    """
    本函数仅完成数据文件分离工作，*.wiped用于下一步挖掘工作,*.dirty用于缺失记录的进一步统计
    文件操作：剔除含有未知属性的记录
    :param fname: 原始数据文件
    """
    # 需要操作的三个文件：原文件，清洗后的文件，脏数据（属性缺失）文件
    file = open(fname, 'r')
    f_wiped = open(fname+'.done','w')
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


def handle_normal(fname:str):
    """
    读取数据文件,完成字符串数据到int数据的映射,对原来的continuous数据不做处理
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

def handle_continuous(data:np.ndarray):
    """
    对数据的连续数据做离散处理
    返回处理过后的ndarray多维数组
    """
    for i in range(data.shape[0]):
        # 1. age-0、hours_pw-12，采用5的区间映射到0到20的取值范围
        data[i,0] = data[i,0]/5
        data[i, 12] = data[i, 12] / 5

        # 2.fnlwgt-2,先取log运算，log后的取值范围[9,15]，再按0.5长度做等区间长度划分
        data[i,2] = (np.log(data[i,2])-9)/0.5

        # 3. cpl_gain-10、cpl_loss-11,简单做标记，对不为0的元素取1做标志
        data[i,10] = 1 if data[i,10]>0 else 0
        data[i, 11] = 1 if data[i, 11] > 0 else 0
    return data

def saveTxt(fname,data:np.ndarray):
    """
    以整数形式，按逗号分割保存数据到文本文件
    :param fname:
    :param data:
    :return:
    """
    np.savetxt(fname, data, fmt='%d', delimiter=',')


def load_np_file(fname):
    """
    读取经过numpy整数化之后输出的数据文件
    :param fname: 文件名
    :return: 数据对应的ndarray类型多维数组
    """
    return np.loadtxt(fname,int,delimiter=',')


#-----------------------------------------------------
def statistics_continuous(data:np.ndarray):
    """
    对数据集的连续属性做统计：最大值，最小值，平均值，无偏样本标准差，相关性，
    :param data:
    :return:
    """
    # 对连续性型数据做统计分析：min、max、mean、std、corf的属性，连续属性不做绘图
    attnamelist = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    print('连续属性的统计分析：')
    column_class = data[:,14]
    for i in range(len(attr_names)-1):
        attr_name = attr_names[i]
        if attr_name in attnamelist:
            column = data[:, i]
            v_min = np.min(column)
            v_max = np.max(column)
            v_mean = np.mean(column)
            v_std = np.sqrt(( column.var() * column.size) / (column.size - 1))
            v_corf = np.corrcoef(column,column_class)
            print(attr_name+':',v_min,v_max,('%.2f' % v_mean),('%.2f' % v_std),('%.3f' % v_corf[0,1]))


def plot_discrete(data:np.ndarray):
    # 对离散型属性做分布统计，各取值的占比以及对类分布情况(二分条形图)
    attnamelist = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country', 'class']
    print('离散属性的统计分析：')
    figure = plt.figure()
    k = 0  # 子图位置标记
    for i in range(len(attr_names) - 1):
        attr_name = attr_names[i]
        if attr_name in attnamelist:
            column = data[:, i]
            # 统计
            print(attr_name + ':', end='')
            unique, index, counts = np.unique(column, return_index=True, return_counts=True)
            for j in range(unique.size):
                print('%d %d %.2f\t' % (unique[j], counts[j], counts[j] / column.size), end='')
            print('')

            # 绘图
            k += 1
            # 记录对应的unique属性值的类标号为1的数量，初始化都为0
            count_class1 = np.zeros(unique.size, int)
            # TODO 这里方法可以改进
            for j in range(data.shape[0]):
                count_class1[list(unique).index(data[j, i])] += data[j, 14]
            count_class2 = counts - count_class1

            axe = figure.add_subplot(330 + k)
            axe.bar(unique, count_class1, color='b', label='>50k', alpha=0.5)
            axe.bar(unique, count_class2, color='r', label='<=50k', bottom=count_class1, alpha=0.5)
            axe.legend(loc=0)
            axe.set_xlabel(attr_name)
            # axe.set_ylabel(attr_name)
            axe.set_yticks([])
            axe.set_xticks([])
            # axe.set_title(attr_name)
    plt.show()


def plot_3d_scatter(data:np.ndarray):
    """
    根据3个与class相关度最高的属性，绘制三维散点图
    三个连续属性：age-0、edu_num-4、hours_pw-12（cpl_gain-10）
    三个字符串属性：occupation-6、marital_status-5、education-3
    """
    d0 = data[np.where(data[:, 14] == 0)]
    d1 = data[np.where(data[:, 14] == 1)]

    # 图一、连续属性的散点图
    figure = plt.figure()
    axe = figure.add_subplot(111, projection='3d')
    axe.scatter(d0[::20,0], d0[::20,4], d0[::20,12], c='r', marker='.')
    axe.scatter(d1[::20,0], d1[::20,4], d1[::20,12], c='g', marker='x')
    axe.set_xlabel(attr_names[0])
    axe.set_ylabel(attr_names[4])
    axe.set_zlabel(attr_names[12])
    plt.show()

    # 图二、字符串属性的散点图
    figure = plt.figure()
    axe = figure.add_subplot(111, projection='3d')
    axe.scatter(d0[::20,3], d0[::20,5], d0[::20,6], c='r', marker='.')
    axe.scatter(d1[::20,3], d1[::20,5], d1[::20,6], c='g', marker='x')
    axe.set_xlabel(attr_names[3])
    axe.set_ylabel(attr_names[5])
    axe.set_zlabel(attr_names[6])
    plt.show()

    # 图三，混合：attr 0 4 5
    figure = plt.figure()
    axe = figure.add_subplot(111, projection='3d')
    axe.scatter(d0[::20, 0], d0[::20, 5], d0[::20, 4], c='r', marker='.')
    axe.scatter(d1[::20, 0], d1[::20, 5], d1[::20, 4], c='g', marker='x')
    axe.set_xlabel(attr_names[0])
    axe.set_ylabel(attr_names[5])
    axe.set_zlabel(attr_names[4])
    plt.show()
#-----------------------------------------------

if __name__ == '__main__':
    """
    处理顺序：（结果：一个最终数据文件，一个统计表格，两张图片（bar、3d_scatter））
    1.remove_unknow_records(fname:str)处理原数据文件，得到清洗缺失属性的记录后的有效数据文件*.wiped，
        同时产生缺失属性记录的数据集*.dirty
    2.load_data(fname:str)读取*.wiped数据文件，对数据中的字符串内容做int映射，得到一个numpy的多维数组ndarray
    4.statistics_continuous(data:np.ndarray)，连续属性做统计分析
    5.plot_discrete(data:np.ndarray)，离散的字符串属性做二分条形图，stacked bar
    6.mapping(data:np.ndarray),对连续型数据做离散映射
    7.plot_3d_scatter(data:np.ndarray):，做3维散点图，简单得查看class的空间分布与属性的关系
    8.saveTxt(fname,data:np.ndarray)与load_np_file(fname)两个方法实现ndarray的保存与读取。
    """
    filename = 'adult.data'
    # 1. 去除不全的数据，并保存文件
    discard_records(filename)
    # 2. 读取去除之后的所有的完整数据集，同时对字符串数据做映射
    data = handle_normal(filename+'.done')
    # 3. 保存str映射之后的数据到.continuous(表示该数据文件中连续属性未处理)
    #saveTxt('adult.continuous',data)
    # 4. 加载数据
    #data = load_np_file('adult.continuous')
    # 5. 字符串属性（离散）作图与统计分析（各属性取各值的类占比情况）
    #plot_discrete(data)
    #plot_3d_scatter(data) # 这里3D散点的是原始连续数据的属性
    # 6. 对原始的连续属性做统计
    #statistics_continuous(data)
    # 7. 对连续属性做离散映射
    handle_continuous(data)
    # 8. 对连续属性离散映射之后再做统计
    #statistics_continuous(data)
    # 9. 3D散点图
    #plot_3d_scatter(data) # 这里3D散点的是离散化后的属性
    # 10. 保存数据
    saveTxt(filename+'.done', data)

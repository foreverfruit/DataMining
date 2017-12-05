import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# 数据映射字典
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


def discard(filename:str):
    '''
    step 1 剔除属性缺失的项，分别保存到两个文件中
    :param filename:文件名
    '''
    # 0.处理中需要用到的三个文本，原数据文件、剔除的数据、处理过的数据
    file = open(filename, 'r')
    f_discard = open(filename + '.discard', 'w')
    f_done = open(filename + '.done', 'w')

    # 1.加载数据 & 2.剔除缺失项
    total, count = 0, 0
    line = file.readline()
    while line:
        total += 1
        if line.__contains__('?'):
            count += 1
            f_discard.write(line)
        else:
            f_done.write(line)
        line = file.readline()

    f_discard.write('缺失属性记录数：' + str(count) + '，占比：' + str(count / total))

    file.close()
    f_discard.close()
    f_done.close()


def handle_normal(filename:str):
    '''
    step 2：处理标称属性，处理方法：以类1（>=50k）的概率作为映射值，这样可以衡量不同标称属性值的距离
    workclass-1, education-3, marital-5, occupation-6, relationship-7, race-8, sex-9, country-13, class-14
    :param filename: 文件名
    :return 返回标称属性处理之后的ndarray类型的数据矩阵
    '''
    filename = filename+'.done'
    # 1.为简化运算，先对标称属性取值做int映射，这里无实际意义，仅为了统计方便
    cvt1 = lambda s: dic_workclass.get(str(s, 'utf-8'))
    cvt3 = lambda s: dic_education.get(str(s, 'utf-8'))
    cvt5 = lambda s: dic_marital_status.get(str(s, 'utf-8'))
    cvt6 = lambda s: dic_occupation.get(str(s, 'utf-8'))
    cvt7 = lambda s: dic_relationship.get(str(s, 'utf-8'))
    cvt8 = lambda s: dic_race.get(str(s, 'utf-8'))
    cvt9 = lambda s: dic_sex.get(str(s, 'utf-8'))
    cvt13 = lambda s: dic_native_country.get(str(s, 'utf-8'))
    cvt14 = lambda s: dic_class.get(str(s, 'utf-8'))

    data = np.loadtxt(fname=filename, delimiter=', ', usecols=range(15), dtype=float,
                      converters={1: cvt1, 3: cvt3, 5: cvt5, 6: cvt6, 7: cvt7, 8: cvt8, 9: cvt9, 13: cvt13, 14: cvt14})

    # 2.统计标称属性的每个值的频次
    normal_attr_names = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex','native-country']

    # 所有类标号为1的行向量组成一个新的矩阵用于下文计算频率
    data_class1 = data[data[:,14]==1,:]

    print('---------标称属性-映射关系：------------')
    # 遍历所有属性
    for i in range(len(attr_names)):
        # 只取标称属性做处理
        attr_name = attr_names[i]
        if attr_name in normal_attr_names:
            print(attr_name,'------------------')
            # 选取该属性维度（列）
            column = data[:, i]
            column_class1 = data_class1[:,i]
            # unique返回该属性的value，及各value对应的counts
            values, counts = np.unique(column, return_index=False, return_counts=True)
            # 映射表，key为属性值value，value为该属性值对应的频率freq，最终用freq替代matrix中的value
            mapping = {}
            # 遍历计算每一个value的类标号为1的频率
            for index,v in enumerate(values):
                # 分别计算：value，value对应的count，以及类标号为1的value的count
                value = values[index]
                count_all = counts[index]
                count_class1 = column_class1.tolist().count(value)
                freq = count_class1/count_all
                # 添加该value的映射
                mapping[value] = freq

            # 通过映射表将所有value的值替换为freq
            for index,line in enumerate(data):
                data[index,i] = mapping.get(line[i])

            print(mapping)

    return data

def handle_continuous(data:np.ndarray,filename:str):
    '''
    step 3：处理连续属性
    age-0,fnlwgt-2,edu_num-4,cpl_gain-10,cpl_loss-11,hours_pw-12
    处理方法如下:
    1.归一化到（0,1）区间，
    2.对cpl_gain和cpl_class采用ratio合并属性
    3.对fnlwgt先取log再做归一化映射
    :param data: 标称属性处理过的data矩阵
    :return 返回预处理完成的ndarray类型的数据矩阵
    '''
    # TODO 未考虑实际测试值超出训练集的max和min的情况,处理的时候越界的直接取0或1

    # 第2列先做log变换，10和11列得到ratio存于10列，11列标记为-1，不再使用
    for i in range(data.shape[0]):
        data[i, 10] = data[i, 11] / (data[i, 10] + 1)
        data[i, 11] = -1
        data[i, 2] = np.log(data[i, 2])

    # 得出各列的max和min，用于归一化
    print('----------连续属性最大值，最小值----------')
    print('age',data[:,0].min(),data[:,0].max())
    print('fnlwgt(loged)',data[:,2].min(),data[:,2].max())
    print('edu_num',data[:,4].min(),data[:,4].max())
    print('cpl_loss/gain',data[:,10].min(),data[:,10].max())
    print('hours_pw',data[:,12].min(),data[:,12].max())

    for i in range(data.shape[0]):
        # age，年龄（17,90），直接除以100
        data[i,0] = data[i,0]/100
        # hours_pw，（1,99），直接除以100
        data[i, 12] = data[i, 12]/100
        # edu_num，（1,16），
        data[i, 4] = (data[i, 4]-1) / 15
        # fnlwgt-2,log后的值域为(9.5,14.2),取整到（9,15）
        data[i,2] = (data[i,2]-9)/6
        # cpl_loss/gain,(0,4356)
        data[i,10] = data[i,10]/4356

    # 保存文件,精度取到小数点后5位
    np.savetxt(filename+'.done', data, fmt='%.5f', delimiter=',')
    return data


def statistics_plot(data:np.ndarray):
    # 统计各属性与class的相关度
    print('------------相关度-----------')
    column_class = data[:, 14]
    for i in range(len(attr_names) - 1):
        # 跳过cpl_loss
        if i == 11:
            continue

        attr_name = attr_names[i]
        v_corf = np.corrcoef(data[:, i], column_class)
        print(attr_name + ':', ('%.3f' % v_corf[0, 1]))

    print('-----------------------')
    # 取相关度最高的三个维度，marital-status-5、relationship-7、education-3画3d图
    d0 = data[np.where(data[:, 14] == 0)]
    d1 = data[np.where(data[:, 14] == 1)]

    figure = plt.figure()
    axe = figure.add_subplot(111, projection='3d')
    # 这个20是step步进，是一个抽样操作，否则3d图中点太多，观察不便
    axe.scatter(d0[::20, 3], d0[::20, 5], d0[::20, 7], c='r', marker='.')
    axe.scatter(d1[::20, 3], d1[::20, 5], d1[::20, 7], c='g', marker='x')
    axe.set_xlabel(attr_names[3])
    axe.set_ylabel(attr_names[5])
    axe.set_zlabel(attr_names[7])
    plt.show()


if __name__ == '__main__':
    '''
    数据预处理，包含以下几个步骤：
    1.剔除属性缺失项（到一个单独文本备用）
    2.处理标称属性——以概率做关于距离的映射
    3.处理连续属性——维规约+归一化
    4.保存数据文件
    '''
    filename = 'adult.data'
    discard(filename)
    data = handle_normal(filename)
    data = handle_continuous(data,filename)
    statistics_plot(data)


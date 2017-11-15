import numpy as np
import matplotlib.pyplot as plt

def where_demo():
    # 测试函数，测试where的用法，用于更方便的从多维数组中按条件筛选向量
    a = np.array(range(0,100,1)).reshape((20,5))
    row =  [90,0,0,0,94]

    #print(a)
    #print(row)

    # TODO 用where方法改进
    for i in range(a.shape[0]):
        if a[i,0]==row[0] and a[i,4]==row[4]:
            print(i,a[i])

    # where一次只能判断一个属性条件，不能多个条件并列筛选
    r = np.where(a[:,0]==row[0])
    print(a[np.where(a[:,0]==0)])


def find_percentile(cpl_gain,cpl_loss):
    # 求百分位数
    sorted_cpl_gain = np.sort(cpl_gain)
    gain_5 = sorted_cpl_gain[int(0.05 * sorted_cpl_gain.size)]
    gain_95 = sorted_cpl_gain[int(0.95 * sorted_cpl_gain.size)]
    print('cpl_gain:',gain_5,gain_95)

    sorted_cpl_loss = np.sort(cpl_loss)
    loss_5 = sorted_cpl_loss[int(0.05 * sorted_cpl_loss.size)]
    loss_95 = sorted_cpl_loss[int(0.95 * sorted_cpl_loss.size)]
    print('cpl_loss:',loss_5,loss_95)

    figure = plt.figure()
    ax1 = figure.add_subplot(121)
    ax1.plot(sorted_cpl_gain)
    ax1.set_title('cpl_gain')

    ax2 = figure.add_subplot(122)
    ax2.plot(sorted_cpl_loss)
    ax2.set_title('cpl_loss')

    plt.show()
    '''
    print数据发现，百分位数处理效果不好，loss_5和loss_95都是0,没有起到作用
    通过plot发现，百分位数很受数据规模和分布的影响，数据大部分堆积到某一边缘小区间时影响了它的筛选效果
    '''

def find_Zquantile(cpl_gain,cpl_loss):
    """
    求z分位数，绝对值<=3的认为正常数据，超过的认为离群点
    """
    # 1.sort
    sorted_cpl_gain = np.sort(cpl_gain)
    sorted_cpl_loss = np.sort(cpl_loss)

    # 2.fin mean\std for operation
    m = sorted_cpl_gain.mean()
    s = sorted_cpl_gain.std()
    # 3. operation z quantile
    cpl_gain_2 = (sorted_cpl_gain-m)/s
    # 4. find index of the first element by (z quantile > 3)
    index_up,index_down = 0,0
    for i,e in enumerate(cpl_gain_2):
        if e<-0.2:
            index_down=i
        elif e>0:
            index_up=i
            break

    # 5. find value of element in index_up and index_down as border
    print(index_down,index_up)
    print(sorted_cpl_gain[index_down],sorted_cpl_gain[index_up])
    print(sorted_cpl_gain[index_down:index_up].size,'%.3f' % (sorted_cpl_gain[index_down:index_up].size/cpl_gain.size))

    # 6. plot
    figure = plt.figure()
    ax1 = figure.add_subplot(111)
    ax1.hist(sorted_cpl_gain[index_down:index_up],bins=100)
    ax1.set_title('cpl_gain')
    plt.show()
    '''
    这种方法也被弃用，因为这个离群的阈值和标准std的大小很有关系，不好取，
    如cpl_gain取threshold为[-0.2,0]任然能满足该区间内数据占比大于90%，
    但没法解释，因为这样是认为超过均值的所有元素都是离群的被舍弃，这不科学。
    '''


def mean_percentile(cpl_gain,cpl_loss,clas):
    """
    筛选均值的上下百分之C的数据作为合理数据，其他为异常数据
    没用的，取0的数据占了0.916，用均值做筛选也任然没有好的效果
    """
    # 求均值，及合理值边界
    mean = cpl_gain.mean()
    border_down = 1#mean*0.5,
    border_up = mean*1.5,
    # 检验合理数据比率，超过95%，认为可行。设置离群率5%
    sorted_cpl_gain = np.sort(cpl_gain)
    index_down = np.where(sorted_cpl_gain>=border_down)[0][0]
    index_up = np.where(sorted_cpl_gain>border_up)[0][0]
    print(mean,index_down,index_up,np.where(cpl_gain==0)[0].size/cpl_gain.size)


if __name__ == '__main__':
    pass
    #fnlwgt, cpl_gain, cpl_loss, clas = np.loadtxt('adult.preprocess', int, delimiter=',', usecols=(2, 10, 11, 14), unpack=True)
    #find_percentile(cpl_gain,cpl_loss)
    #find_Zquantile(cpl_gain,cpl_loss)
    #mean_percentile(cpl_gain,cpl_loss,clas)
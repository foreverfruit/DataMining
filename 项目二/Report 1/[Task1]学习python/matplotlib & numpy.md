[TOC]

---

# Matplotlib初步应用



## 环境搭建

采用anaconda，自带各种科学计算的库，省去自己安装，版本适配的麻烦

---

## Numpy

Numpy是一种开源的扩展的数值计算库。

- ndarray，numpy的array对象。三种方法获得该对象：
  - nd_a = np.array(py_a) # python的数组a通过numpy.array方法转变为ndarray类型的数组
  - nd_b = np.arange(10) # arange函数生成ndarray对象
  - 读取数据库或文件：nd_c = np.loadtxt(parameters)，注意参数的作用。
- 索引和切片
  - ndarray和array一样，从0元素开始索引
  - 用name[start :end :step]方式切片
- numpy常用函数
  - min、max、median、mean（平均值）、var（方差）、sort

  - 函数的调用支持面向过程和面向对象两种方式，np.sort(ndarrayObj)和ndarrayObj.sort()效果一样，这里注意一个区别。

  - np.sort(ndarray)不会改变ndarray原数组对象，返回一个新的已排序的对象，而ndarray.sort()不会产生新对象，而是将自身排序。

  - random随机数生成、linspace(start,end,amount)生成等区间的离散列。

  - random.randn(count)生成count个均值为0方差为1的数组

  - 数据的导入：converters参数的使用。

    ```python
    # 这里将第0列日期数据的str型数据转换成float，且日期数据格式为‘12/01/1995’（1995-12-01）
    # 需要导入包
    import matplotlib.datas as matdata
    date,otherdata = np.loadtxt(..,converters={0:matdata.strpdata2num('%m/%d/%y')},..)
    # 显示的时候如果需要这个日期数据，用相应的画图函数
    plt.plot_date(date,otherdata)
    # 这个函数会识别float形式的日期数据

    # 生成数据
    #创造一组服从正态分布的定量数据
    data = normal(0, 10, size=10)
    #创造一组服从均匀分布的定性数据
    data = randint(0, 10, size=10)
    ```

  - numpy统计函数：

    - 位置：均值mean(ndarray)、中位数median(ndarray)、众数mode(ndarray)

    - 发散程度：极差ptp(data)、方差var(ndarray)、标准差std(ndarray)

    - 异常（偏差程度，z-分数）：(data[0]-mean(data)) / std(data)，通常绝对值大于3视为异常

    - 相关度（协方差COV、相关系数衡量CORRCOEF）：

      ![img](http://images2015.cnblogs.com/blog/927391/201604/927391-20160428164241783-360115652.png)

      ```python
      #计算两组数的协方差
      #参数bias=1表示结果需要除以N，否则只计算了分子部分
      #返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的协方差。对角线为方差
      cov(data, bias=1)

      #计算两组数的相关系数
      #返回结果矩阵，第i行第j列的数据表示第i组数与第j组数的相关系数。对角线为1
      corrcoef(data)
      ```


      | 包            | 方法       | 说明             |
      | ------------ | -------- | -------------- |
      | numpy        | array    | 创造一组数          |
      | numpy.random | normal   | 创造一组服从正态分布的定量数 |
      | numpy.random | randint  | 创造一组服从均匀分布的定性数 |
      | numpy        | mean     | 计算均值           |
      | numpy        | median   | 计算中位数          |
      | scipy.stats  | mode     | 计算众数           |
      | numpy        | ptp      | 计算极差           |
      | numpy        | var      | 计算方差           |
      | numpy        | std      | 计算标准差          |
      | numpy        | cov      | 计算协方差          |
      | numpy        | corrcoef | 计算相关系数         |

练习一：numpy生成100以内随机数组，存储数组到文件，读取文件，数组排序、求最大值、最小值、均值、方差

```python
import  numpy as np
# 产生100个元素的ndarray，[0,1000)之间
nd_a = np.random.randint(0,1000,100)

# 保存为文件，参数看文档
np.savetxt('arraydata.txt',nd_a,delimiter=',',header='my random data',footer='over',fmt='%d')

# 读取文件
nd_b = np.loadtxt('arraydata.txt',dtype=int,delimiter=',',skiprows=1,unpack=False,)

# 对比原数组和读取的数组元素个数是否一致
print('loaded-' + str(nd_b.size) + ',original-'+str(nd_a.__len__()))

# 排序
nd_c = np.sort(nd_b)

# 输出排序后的结果：排序产生的新数组和原数组
print(nd_c)
print(nd_b)

# 输出max
print('max:loaded-'+ str(np.max(nd_b)) + ',original-' + str(nd_a.max()))
# 输出min
print('min:loaded-'+ str(np.min(nd_b)) + ',original-' + str(nd_a.min()))
# 输出mean
print('mean:loaded-'+ str(np.mean(nd_b)) + ',original-' + str(nd_a.mean()))
#输出方差var
print('avriance:loaded-'+ str(np.var(nd_b)) + ',original-' + str(nd_a.var()))
```

练习二：Iris数据集的简单练习

```python
import  numpy as np
# 这里的路径是因为本py文件和数据文件不在一个目录下
import  os
pro_root = os.path.abspath('..')
sepal_length,sepal_width,petal_length,petal_width = np.loadtxt(pro_root+'\\dataset\\iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))

# 三种花的统计值，简单的统计
print(sepal_length[:50].mean())
print(sepal_length[:50].var())

print(sepal_length[51:100].mean())
print(sepal_length[51:100].var())

print(sepal_length[101:150].mean())
print(sepal_length[101:150].var())
```



```python
#---------------------numpy简单学习--------------------------------

# 列表list转矩阵
array = np.array([[1,2,3],[2,3,4]]) # 2*3矩阵

# 查看维度
array.ndim # dimention

# 形状，行列值
array.shape

# 大小，个数
array.size

# 指定array类型
x = np.array(list,dtype=np.xxx) # np.xxx表示np的类型
x.dtype # 查看类型

# 生成指定维度的空array
x = np.zeros((2,3,4)) # 这表明x是三维数组，2行3列，第三个维度size(深度、高度)为4

# 同理生成值都为1的array
x = np.ones((2,3,4))   # 同理，np.empty((shape)) 生成近似zeros的array

# 产生随机的矩阵
x = np.random.random((shape)) # 生成shape样式的矩阵，值随机(0,1)区间内

# 生成有序的序列
x = np.arange(start,end,step)  # 生成1维数据列，类似python的range的效果，默认区间左闭右开
# 注，range返回的是range对象，到list还需要转换

# 将np.array重新构形
x.reshape((new shape)) # 比如 new shape = (2,3,4) 将x装换为3维矩阵
# 注，转换前后的size不能变，如size=10的array不能reshape为(2,3,4)的矩阵

x.reshape((-1,3,4)) # new shape中可以用-1表示自动计算值，根据size自动填写，显然只能指定一个-1值

# 区间均分,类似于分箱
x = np.linspace(start,end,nbins) # 将[start,end]区域分成nbins段，返回一个array，默认包含首尾（闭区间）


# -------------- 基础运算 -------------------------------
# np中的运算会作用于array的每个元素,可以直接调用np的函数，也可以用oob方式调用ndarray的成员方法，且可以指定参数axis，表示对某一维度上进行运算

# 矩阵乘法 np.dot(x,y)
x = np.arange(0,4).reshape((2,2))
y = np.arange(5,9).reshape((-1,2))
r1 = x*y		 # 元素对应相乘
r2 = np.dot(x,y) # 矩阵乘法运算
r2 = x.dot(b)    # 同上

# 最大，最小，求和，一下方法可指定参数axis（如，axis=0），表示在这个维度、轴、rank上运算
np.sum(x) # x.sum()
np.min(x) # x.min()
np.max(x) # x.max()

# 最大值最小值的索引,多维矩阵也按一维线性索引返回
index = np.argmin(x)
index = np.ragmax(x)

# 中位数、平均值、累计值
mean = np.mean(x)
median = np.median(x)
cumsum = np.cumsum(x) # 累计和
np.diff(x) # x[j,i+1]-x[j,i]，按行输出

np.sort(x) # 逐行排序

# 矩阵转置
np.transpose(x) 
x.T

# clip可以理解为一种过滤
np.clip(x,a,b) # x矩阵的元素中所有小于a的元素置为a,大于b的置为b,即ab为取值范围

# 多维矩阵的单行形式
x.flat # x.flatten() ，当然也可以reshape到一维数组

# 数据合并
np.vstack((a,b)) # vertical 垂直合并
np.hstack((a,b)) # horizontal 水平合并a、b
np.concatenate((a,b,c,d,...,n),axis=j) # 在j维度上对a到n的array合并

# 分裂矩阵
np.split(x,nbins,axis=n) # 对x做第n维的nbins等份分割，检测nbins必须满足axis=n上等份要求，否则出错，如4列分3份，出错……
np.array_split(x,nbins,axis=n) # 不等份 分割。对nbins无检测

##！ python默认是引用赋值
a = np.array(....)
b = a # 此时a is b == True，同一块内存，修改一个，都改变
b = a.copy() # 重新创建一个值一样的对象给b，此时a、b两个不同内存空间的对象，不再关联，彼此独立
```



---
## 散点图

scatter(xList,yList)函数

外观参数：用以区分不同类型数据。颜色-c，点大小-s，透明度-alpha，点形状-marker

```python
# 查看花的sepal的length和width的关系
import  numpy as np
import  matplotlib.pyplot as plt
import  os
pro_root = os.path.abspath('..')

sepal_length,sepal_width = np.loadtxt(pro_root+'\\dataset\\iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1))

plt.scatter(sepal_width[:50],sepal_length[:50],marker='*',c='red',alpha=0.5,s=50)
plt.scatter(sepal_width[51:100],sepal_length[51:100],marker=',',c='green',alpha=0.5,s=50)
plt.scatter(sepal_width[101:150],sepal_length[101:150],marker='.',c='blue',alpha=0.5,s=50)
plt.show()

# 可以看到三种花的sepal的宽和长都是近似正相关的
```

---

## 折线图

plt.plot(xlist,ylist)、plt.plot_date(date,data,linestyle='.')(这个函数需要指定linestyle，否则画出的是点，而不是线)

练习一：[0,10]内的正弦图象

```python
# 0,10内正弦图象
import  numpy as np
import  matplotlib.pyplot as plt

# 只分成20份，能看到折线
xlist = np.linspace(0,10,20)
ylist = np.sin(xlist)
plt.plot(xlist,ylist,'b:')

# 分成100份，几乎是光滑曲线
alist = np.linspace(0,10,100)
blist = np.cos(alist)
plt.plot(alist,blist,'r--')

plt.show()

# plot函数的参数需要注意,'r--'这是一种组合的参数，r表示颜色red，--表示线的样式，具体的组合看文档
```

练习二：iris练习

```python
import  numpy as np
import  matplotlib.pyplot as plt
import  os
pro_root = os.path.abspath('..')
data = np.loadtxt(pro_root+'\\dataset\\iris.data',
                  delimiter=',',
                  unpack=False,dtype=float,usecols=(0,1,2,3))

for x in data[0:50]:
    plt.plot(x,'r--')
for x in data[51:100]:
    plt.plot(x,'g-')
for x in data[100:150]:
    plt.plot(x,'b-.')

# 对图做个加工
# x、y轴标签与图形标题
plt.xlabel('category')
plt.ylabel('value')
plt.title('iris\'s sepal and petal leagth and width')

# X轴上的标签
xticks=['sepal length','sepal width','petal length','petal width']
#设置x轴的刻度，将构建的xticks代入
plt.xticks(range(len(xticks)),xticks,rotation=0)

# 图中可以看出三种类型的折线图
plt.show()

```

---

## 条形图

np.bar(parameters)

练习：iris的两种条形图

```python
"""
累计条形图，stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import utils as tool

sepal_length,sepal_width,petal_length,petal_width = np.loadtxt(tool.getPath()+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))

# data
sun_sepal_length = [np.sum(sepal_length[0:50]),np.sum(sepal_length[51:100]),np.sum(sepal_length[101:150])]
sun_sepal_width = [np.sum(sepal_width[0:50]),np.sum(sepal_width[51:100]),np.sum(sepal_width[101:150])]
sun_petal_length = [np.sum(petal_length[0:50]),np.sum(petal_length[51:100]),np.sum(petal_length[101:150])]
sun_petal_width = [np.sum(petal_width[0:50]),np.sum(petal_width[51:100]),np.sum(petal_width[101:150])]


# plot
left = np.arange(3)
width = 0.5
# 这里有大量重复的矩阵计算，需要优化
bar1 = plt.bar(left,sun_sepal_length,width,yerr=sun_sepal_length,align='center',color='red')
bar2 = plt.bar(left,sun_sepal_width,width,yerr=sun_sepal_width,bottom=sun_sepal_length,align='center',color='green')
bar3 = plt.bar(left,sun_petal_length,width,yerr=sun_petal_length,bottom=np.add(sun_sepal_length,sun_sepal_width),align='center',color='blue')
bar4 = plt.bar(left,sun_petal_width,width,yerr=sun_petal_width,bottom=np.add(np.add(sun_sepal_length,sun_sepal_width),sun_petal_length),align='center',color='black')
# 坐标轴美化
plt.ylabel('value')
plt.title('this is title')
plt.xticks(left,('typeA','typeB','typeC'))
# 图例标注
plt.legend((bar1[0], bar2[0],bar3[0],bar4[0]), ('sepal_length', 'sepal_width','petal_length','petal_width'))
plt.show()
```

```python
"""
并列对比条形图
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import utils as tool

sepal_length,sepal_width,petal_length,petal_width = np.loadtxt(tool.getPath()+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))

# data
sun_sepal_length = [np.sum(sepal_length[0:50]),np.sum(sepal_length[51:100]),np.sum(sepal_length[101:150])]
sun_sepal_width = [np.sum(sepal_width[0:50]),np.sum(sepal_width[51:100]),np.sum(sepal_width[101:150])]
sun_petal_length = [np.sum(petal_length[0:50]),np.sum(petal_length[51:100]),np.sum(petal_length[101:150])]
sun_petal_width = [np.sum(petal_width[0:50]),np.sum(petal_width[51:100]),np.sum(petal_width[101:150])]

x = np.arange(0,15,5)
width = 0.5

bar1 = plt.bar(x,sun_sepal_length,width,color='red')
bar2 = plt.bar(x+width+0.1,sun_sepal_width,width,color='green')
bar3 = plt.bar(x+width*2+0.2,sun_petal_length,width,color='blue')
bar4 = plt.bar(x+width*3+0.3,sun_petal_width,width,color='black')

plt.xticks(x+width*2,('A','B','C'))
plt.legend((bar1[0],bar2[0],bar3[0],bar4[0]),('sepal_length','sepal_width','petal_length','petal_width'),loc=2)

plt.show()
```

---

## 直方图

类似于条形图，但是通常是展示连续数据。

plt.hist(data,bins,color,normed)，bins表示的直方图的分段数量，normed决定y轴是频率还是计数

```python
# 随机生成数据做直方图
import numpy as np
import matplotlib.pyplot as plt

mean = 20
var = 0.5
data = mean+var*np.random.randn(100000)

# plt.hist(data,bins=50,normed=False,color='red')
# 当直方图的划分很大时，就是密度曲线
plt.hist(data,bins=500,normed=True,color='red')
plt.show()
```

```python
# 二维的直方图，用颜色表示联合密度
import numpy as np
import matplotlib.pyplot as plt

# xy均值分别为1和5，方差都为1
x = 1+np.random.randn(2000)
y = 5+np.random.randn(2000)

plt.hist2d(x,y,bins=40)
# 图通过颜色深浅表示概率，显然在(x,y)=(1,5)处密度最大，颜色最亮
plt.show()
```

---

## 饼状图

plt.pie(dataArray,labels,autopct='%d',explode,shadow)

- autopct用于格式化饼块的占比数据显示格式
- explode是一个集合，表示每一块饼距离中心的距离，用以强调突出某一个饼块
- shadow是boolean型数据指示是否给饼块添加立体阴影

```python
import matplotlib.pyplot as plt

data = (20, 10, 30, 25)
labels = 'SH', 'BJ', 'SZ', 'GZ'
colors = ('red', 'green', 'blue', 'yellow')
explode = (0, 0, 0.2, 0.05)

# 饼图中坐标x、y默认不是1比1的，所以图像会椭圆，需要指定坐标轴1:1以画出正圆
plt.axes(aspect=1)

plt.pie(x=data,labels=labels,colors=colors,explode=explode, autopct='%0.1f%%', shadow=True)
plt.show()

```

---

## 箱形图

plt.boxplot(data,sym,whis)

- sym表示异常点的显示样式，可同时指定颜色、样式等
- whis，whiskers，表示离群点的范围。箱形图会统计[最小值、四分位值Q1、中间值、四分之三分位值Q3、最大值，平均值，以及离群点]，这里最小值和最大值是排除离群点的，那么怎么确定离群点和最小值最大值之间的区分呢？用whis，$最小值=Q1-(Q3-Q1)*whis$，$最大值=Q3+(Q3-Q1)*whis$，即异常点的判定取决于箱长(Q3-Q1)和whis。whis默认为1.5

```python
import numpy as np
import matplotlib.pyplot as plt
from src.utils import utils as tool

sepal_length,sepal_width,petal_length,petal_width = np.loadtxt(tool.getPath()+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))

# 画三种花的sepal_length的箱形图
x = sepal_length[:50],sepal_length[51:100],sepal_length[101:150]
labels = 'A','B','C'

plt.boxplot(x,labels=labels,whis=1,sym='bx')
plt.show()
```

```python
# test whis
import matplotlib.pyplot as plt

x = range(1,100,1)
# Q1=25 MID=50 Q3=75 boxLength=50, 如果whiskers=0.4,则上下都会有5个离群点
plt.boxplot(x,whis=0.4,sym='r.')
plt.show()
```

---

## 颜色和样式

调整颜色、线型、点型

- 字符表示八种内建颜色：red、green、blue、cyan、magenta、yellow、black、white，均可用首字母替代。还可以十六进制表示颜色，如#FF08AB，分别表示三个颜色通道值。还可以RGB元组表示，如（255,255,255）。还有一种灰度表示，color=‘0.5’，表示图像是一个灰色图像，值表示颜色程度。
- 点，直接在matplotlib的官网文档中找markers。matplotlib默认在显示不同marker时分配不同颜色（认为是不同的需要显示的对象）。
- 线：‘--‘虚线，’-‘实线，’-.‘点划线，’:‘点线。
- 样式字符串：可同时在字符串中包含以上三种样式信息，如'rx--'，表示红色，点为x，线为虚线

---

## 编程方式

pyplot、pylab和面向对象的方式

- pyplot：简单易用，底层定制能力不够。
- pylab：结合matplotlib和numpy的模块，模拟了matlab的环境，不推荐用了
- 面向对象：matplotlib的精髓，更基础和底层的方式。

推荐13两种方式根据实际情况综合使用

---

## 子图subplot

matplotlib绘图的三个层级：FigureCanvas 画布，Figure 图，Axes 坐标系。一张图上画多个对比图像，这其实不是多个Figure，而是一个Figure上建立多个Axes，每个Axes表达一个内容。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,10)

fig = plt.figure()

axe1 = fig.add_subplot(221)
axe2 = fig.add_subplot(222)
axe3 = fig.add_subplot(223)
axe4 = fig.add_subplot(224)

axe1.plot(x,[ i**3 for i in x])
axe2.scatter(x,x)
axe3.hist(x,bins=9)
axe4.plot(x,-x)

plt.show()
#--------------另一种方式----------------
fig, axes = plt.subplots(ncol=2,nrow=2)
ax1,ax2,ax3,ax4 = axes.ravel()
# 再用每个ax再每个图轴上画数据
```



## 多图

```python
import matplotlib.pyplot as plt

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot([1,2],[1,2])

f2 = plt.figure()
ax2 = f2.add_subplot(222)
ax2.plot([1,2],[2,1])

# 一次弹出两个图框
plt.show()
```



## 网格

axe.grid()、plt.grid()



## 图例

画图的时候指定label，然后plt.legend()自动根据label显示图例

图例可以由plt显示，也可以由axe显示。同样label可以在不同的时机创建

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(x,x,label='$ y = x $',color='b')
ax.plot(x,x**2,label='$ y = x^2 $',color='g')
ax.plot(x,x**3,label='$ y = x^3 $',color='r')

ax.grid(linestyle='-.',color='y')

ax.legend()

# 调整坐标轴范围(x1,x2,y1,y2)
plt.axis([0,10,0,500])
```



## 坐标轴范围

```python
plt.axis([x1,x2,y1,y2])
plt.xlim()
plt.ylim()

```



## 坐标轴刻度

```python
ax = plt.gca() # get current axe,获取当前的坐标轴
ax.locator_params(nbins=10) # nbins表示需要将轴刻度分成几个格子，可以通过'x'/'y'制定某一个坐标轴
```

日期作为轴刻度

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 导入时间模块
import datetime

# 创建两个时间对象
start = datetime.datetime(2015,11,6)
stop = datetime.datetime(2017,11,6)
# 创建时间间隔对象
delta = datetime.timedelta(days=1)
# 创建时间格式化器
date_format = mpl.dates.DateFormatter('%Y-%m-%d')

# 转换成matplotlib的时间序列对象
dates = mpl.dates.drange(start,stop,delta)

# 生成随机数据
y = np.random.randn(len(dates))

ax = plt.gca()
# 画时间折线图，实线，不给marker（不会突出数据点）
ax.plot_date(dates,y,linestyle='-',marker='')
# 将日期格式化对象应用到x轴上
ax.xaxis.set_major_formatter(date_format)

# 调整x轴日期标签自适应显示
plt.gcf().autofmt_xdate()

plt.show()
```



## 添加坐标轴

效果：多重y轴，同一个axe中的两个线采用不用的刻度，这个时候需要两条对应的坐标轴，左右并列分别对应两条数据线。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,11)

y1 = x**2
y2 = -3*x+3

plt.plot(x,y1,color='r',label='$y1=x^2$')

# 添加一条并列坐标轴
plt.twinx()
plt.plot(x,y2,color='b',label='$y2=-3*x+3$')

# 此时两条线y1 y2就在同一个axe上对应不同的y轴
plt.legend()
plt.show()

#-----------------OOP方式--------------------------
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,11)

y1 = x**2
y2 = -3*x+3

fig = plt.figure()
ax = fig.add_subplot(111)

l1, = ax.plot(x,y1,color='r',label='$y1=x^2$')

ax2 = ax.twinx()
l2, = ax2.plot(x,y2,color='b',label='$y2=-3*x+3$')

# 这里需要了解axe和figure的范围，若gcf()再legend，图例会滑到整个图的四个角，坐标轴外面
# 若gca()表示获取到轴，再legend，图例画在轴范围内
plt.gca().legend((l1,l2),('A','B'),loc=0)
plt.show()

#---------------------
# 同理，twiny可以获得并列的x轴共用一个y轴
```



## 注释

key word：annotate

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10,11)
y = x**2

plt.plot(x,y,'b-')

# xy表示箭头的位置，xytext表示注释文本的坐标
# arrowprops字典类型的关于箭头的属性设置。
# facecolor颜色，frac箭头百分百，arrowstyle是一个内置样式，查看文档具体的props设置
plt.annotate('this is bottom',xy=(0,1),xytext=(-3,20),
             arrowprops={'arrowstyle':'->'})

plt.show()

```



## 画文字

plt.text(x,y,words)，words文字内容自动识别Latex形式的字符串

[Latex](http://matplotlib.org/tutorials/text/mathtext.html#sphx-glr-tutorials-text-mathtext-py)学习链接

注：因为latex中有大量的反斜杠，推荐用r'........'的形式创建字符串，表示默认不转义反斜杠



## 区域填充

区域上色

key words: fill()	fill_between()

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5*np.pi,1000)

y1 = np.sin(x)
y2 = np.cos(x)
'''

plt.fill(x,y1,'b',alpha=0.3)
plt.fill(x,y2,'r',alpha=0.3)

'''
# 交集填充，where表示填充区域的条件表达式
# interpolate表示是否完全填充，因为数据点是离散的，粒度太粗的时候，会有区域填充不到，这个参数会实现完全填充
plt.fill_between(x,y1,y2,where=(y1>y2),facecolor='r',alpha=0.5,interpolate=True)
plt.fill_between(x,y1,y2,where=(y1<y2),facecolor='b',alpha=0.5)

plt.show()

```



## 形状

key words: matplotlib.patches		add_patch

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# 构建一个圆对象,xy是圆心位置
circle = mp.Circle(xy=(1,2),radius=1,
                   color='blue',alpha=0.5,fill=True,ls='-.')

# 矩形，xy是左下角坐标
rect = mp.Rectangle(xy=(3,3),width=2,height=1,color='g',ls='-',alpha=0.5)

# 多边形
rp = mp.RegularPolygon(xy=(5,5),numVertices=7,radius=1,alpha=0.3,color='r')

# 椭圆
elp = mp.Ellipse(xy=(1,4),width=3,height=1,color='y',alpha=0.3)

ax = plt.gca()
# 该图轴上添加这个图块，传入图块对象
ax.add_patch(circle)
ax.add_patch(rect)
ax.add_patch(rp)
ax.add_patch(elp)

# 设置轴范围
# ax.set_xlim(0,10)
# ax.set_ylim(0,10)

# 设置坐标轴相等，xy比例相同
plt.axis('equal')
plt.show()
```



## 美化-样式

使用多种画图样式

key words：plt.style	plt.style.use('xx')	plt.style.available



## 极坐标

```python
import matplotlib.pyplot as plt
import numpy as np

r = np.arange(1,6,1)
theta = np.arange(0,2*np.pi+1,np.pi/2)

# projection透射，表示轴的投射方式，这里采用极坐标polar
ax = plt.subplot(111,projection='polar')
ax.plot(theta,r,color='r',linewidth=3)
ax.grid(True)

# 再画一条
r2 = [5,5,5,5,5]
r2 = np.empty(5) # 5个空元素
r2.fill(5) # 元素填充为5

theta2 = [0,np.pi/2,np.pi,np.pi*3/2,0]
ax.plot(theta2,r2,color='g',lw=3)

plt.show()

# --------------------------------
# 当然也可以直接采用plt的方法
# plt.polar()
```



## matplotlib中文显示问题

```python
#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
```



## 练习项目一：函数积分图

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

x = np.linspace(0,10)
y = -(x-2)*(x-8)+40

fig = plt.figure()

ax = fig.add_subplot(111)

# matplotlib提供长参数的简写，linewidth简写为lw，同理linestyle=ls
ax.plot(x,y,'r-',lw=2)

# 设置两个积分起止点
a,b = 2,9

# 调整坐标轴
ax.set_xticks([a,b])              # 设置点，找到点
ax.set_xticklabels(['$a$','$b$'])     # 设置tick的标签
ax.set_yticks([])

# 将右边和上边的框去掉
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(xmin=1)
#ax.xaxis.set_ticks_position('bottom')


# 画坐标轴的显示--这里不太理想，这种方式，可能存在改进
fig.text(0.9,0.05,'x')
fig.text(0.1,0.9,'y')

# 画积分区域，patches，多边形polygon，需要一个二维的数组（x,y）序列作为它的边界
ix = np.linspace(a,b)
iy = -(ix-2)*(ix-8)+40
# 这里构造数组的方法用到了zip函数和list函数
verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
# 构建Polygon类型的patches对象，颜色设置利用[0,1]内的数字表示灰度
poly = Polygon(verts,facecolor='0.9',edgecolor='0.5')

# 画patch
ax.add_patch(poly)

# 画公式
str = r'$\int_a^b(-(x-2)(x+8)+40)dx$'
sx = (a+b)/2
sy = 35
# 对齐方式可用 ha简写代替 horizontalalignment
ax.text(sx,sy,str,fontsize=10,ha='center')

plt.show()
```



## 练习项目二：散点图和直方图的联合

```python
import matplotlib.pyplot as plt
import numpy as np

# 使用其他的画图样式
plt.style.use('ggplot')

# 画图数据（正相关的x和y）
x = np.random.randn(200)
y = x + np.random.randn(200)*0.8

# 定义一些距离：margin_border,width,height,margin_gap
# 分别表示， 图边距，散点图宽(正方形),直方图高，散点图和直方图间距
margin_border,width,height,margin_gap=0.1,0.6,0.2,0.02

# 散点图坐标
s_bottom_x=margin_border
s_bottom_y=margin_border
s_height = s_width = width

# 直方图1坐标
h1_bootom_x = margin_border
h1_bootom_y = s_bottom_y+width+margin_gap
h1_height=height
h1_width = width

# 直方图2坐标
h2_bottom_x = margin_border+width+margin_gap
h2_bottom_y = margin_border
h2_height=width
h2_width=height

# 生成相应的轴
rect = [s_bottom_x,s_bottom_y,s_width,s_height]
ax_scatter = plt.axes(rect)

rect = [h1_bootom_x,h1_bootom_y,h1_width,h1_height]
ax_h1 = plt.axes(rect)

rect = [h2_bottom_x,h2_bottom_y,h2_width,h2_height]
ax_h2 = plt.axes(rect)

# 调整坐标轴，删除一些多余的轴刻度
ax_h1.set_xticks([])
ax_h2.set_yticks([])

# 画图
ax_scatter.scatter(x,y,color='r',alpha=0.5)

# 固定直方图的箱体宽度
bin_width = 0.25
# 计算箱数
maxValue = np.max([np.max(np.fabs(x)),np.max(np.fabs(y))])
nbins = int(maxValue/0.25+1)

# 根据数据范围为散点图设置刻度范围
ax_scatter_lim = nbins*bin_width
ax_scatter.set_xlim(-ax_scatter_lim,ax_scatter_lim)
ax_scatter.set_ylim(-ax_scatter_lim,ax_scatter_lim)

# 画直方图
ax_h1.hist(x,bins=nbins,color='g',alpha=0.5)
ax_h1.set_xlim(ax_scatter.get_xlim())
ax_h1.locator_params('y',nbins=8)

ax_h2.hist(y,bins=nbins,orientation='horizontal',color='b',alpha=0.5)
ax_h2.set_ylim(ax_scatter.get_ylim())
ax_h2.locator_params('x',nbins=8)

plt.show()
```














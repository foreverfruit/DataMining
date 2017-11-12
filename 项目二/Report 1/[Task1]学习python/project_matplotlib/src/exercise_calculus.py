# 绘制积分函数图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

x = np.linspace(0,10)
y = -(x-2)*(x-8)+40

fig = plt.figure()

ax = fig.add_subplot(111)

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
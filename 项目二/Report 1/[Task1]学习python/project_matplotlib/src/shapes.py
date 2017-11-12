# ipython file 用于交互式输入输出
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
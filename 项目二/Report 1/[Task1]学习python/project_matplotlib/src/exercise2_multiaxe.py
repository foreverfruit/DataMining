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
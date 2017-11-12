import matplotlib.pyplot as plt

data = (20, 10, 30, 25)
labels = 'SH', 'BJ', 'SZ', 'GZ'
colors = ('red', 'green', 'blue', 'yellow')
explode = (0, 0, 0.2, 0.05)

# 饼图中坐标x、y默认不是1比1的，所以图像会椭圆，需要指定坐标轴1:1以画出正圆
plt.axes(aspect=1)

plt.pie(x=data,labels=labels,colors=colors,explode=explode, autopct='%0.1f%%', shadow=True)
plt.show()

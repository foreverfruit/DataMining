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
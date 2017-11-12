import matplotlib.pyplot as plt

x = range(1,100,1)
# Q1=25 MID=50 Q3=75 boxLength=50, 如果whiskers=0.4,则上下都会有5个离群点
plt.boxplot(x,whis=0.4,sym='r.')
plt.show()
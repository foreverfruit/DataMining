import  numpy as np
import  matplotlib.pyplot as plt
import  os

# load data
pro_root = os.path.abspath('..')
sl,sw,pl,pw = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))
datarray = np.array([sl,sw,pl,pw])
attr_names = ['sepal length','sepal width','petal length','petal width']
class_names = ['setosa','versicolor','virginica']

x1 = np.linspace(4.3, 7.9,150)
x2 = np.linspace(2.0, 4.4,150)
x3 = np.linspace(1.0, 6.9,150)
x4 = np.linspace(0.1, 2.5,150)
x = (x1,x2,x3,x4)
colors = ('r','g','b','y')

# plot
plt.style.use('ggplot')
figure = plt.figure()
for i,data in enumerate(datarray):
    axe = figure.add_subplot(221+i)
    axe.scatter(x[i],data,color=colors[i],alpha=0.5)

plt.show()
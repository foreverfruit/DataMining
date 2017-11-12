import  numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  os

# load data
pro_root = os.path.abspath('..')
sl,sw,pl,pw = np.loadtxt(pro_root+'/dataset/iris.data',
                  delimiter=',',
                  unpack=True,dtype=float,usecols=(0,1,2,3))
datarray = np.array([sl,sw,pl,pw])
attr_names = ['sepal length','sepal width','petal length','petal width']
class_names = ['setosa','versicolor','virginica']

# plot
plt.style.use('ggplot')
figure = plt.figure()
axe = figure.add_subplot(111, projection = '3d')

axe.scatter(sw[:50], pl[:50], pw[:50], c='r', marker='^')
axe.scatter(sw[50:100], pl[50:100], pw[50:100], c='g', marker='x')
axe.scatter(sw[100:150], pl[100:150], pw[100:150], c='b', marker='.')

axe.set_xlabel(attr_names[1])
axe.set_ylabel(attr_names[2])
axe.set_zlabel(attr_names[3])

plt.show()
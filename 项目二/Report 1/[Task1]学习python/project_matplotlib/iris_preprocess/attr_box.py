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

# plot
plt.style.use('ggplot')
figure = plt.figure()
for i,data in enumerate(datarray):
    d1,d2,d3 = data[:50],data[50:100],data[100:]
    axe = figure.add_subplot(221+i)
    axe.boxplot((d1,d2,d3), whis=0.9, sym='rx')
    axe.set_title(attr_names[i])

plt.show()
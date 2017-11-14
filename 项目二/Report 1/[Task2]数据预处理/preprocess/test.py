import numpy as np

a = np.array(range(0,100,1)).reshape((-1,5))
row =  [90,0,0,0,94]

# TODO 用where方法改进
for i in range(a.shape[0]):
    if a[i,0]==row[0] and a[i,4]==row[4]:
        print(i,a[i])
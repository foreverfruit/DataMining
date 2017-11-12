# numpy生成100以内随机数组，存储数组到文件，读取文件，数组排序、求最大值、最小值、均值、方差
import  numpy as np
import  os
# 产生100个元素的ndarray，[0,1000)之间
nd_a = np.random.randint(0,1000,100)
pro_root = os.path.abspath('..')
# 保存为文件，参数看文档
np.savetxt(pro_root + '/dataset/array.txt',
           nd_a,delimiter=',',
           header='my random data',
           footer='over',fmt='%d')

# 读取文件
nd_b = np.loadtxt(pro_root+'/dataset/array.txt',
                  dtype=int,
                  delimiter=',',
                  skiprows=1,unpack=False,)

# 对比原数组和读取的数组元素个数是否一致
print('loaded-' + str(nd_b.size) + ',original-'+str(nd_a.__len__()))

# 排序
nd_c = np.sort(nd_b)

# 输出排序后的结果：排序产生的新数组和原数组
print(nd_c)
print(nd_b)

# 输出max
print('max:loaded-'+ str(np.max(nd_b)) + ',original-' + str(nd_a.max()))
# 输出min
print('min:loaded-'+ str(np.min(nd_b)) + ',original-' + str(nd_a.min()))
# 输出mean
print('mean:loaded-'+ str(np.mean(nd_b)) + ',original-' + str(nd_a.mean()))
#输出方差var
print('avriance:loaded-'+ str(np.var(nd_b)) + ',original-' + str(nd_a.var()))
[TOC]

## Python基础

内置函数type()，查询变量类型

print的另一个用法，print后跟多个输出，以逗号分隔。

序列有两种：tuple（定值表； 也有翻译为元组） 和 list (表)。tuple和list的主要区别在于，一旦建立，tuple的各个元素不可再变更，而list的各个元素可以再变更。序列元素的下标从0开始，元素类型可以不同。

字符串是一种特殊的元组。

print(5 in [1,3,5]）       # 5是list [1,3,5]的一个元素，打印布尔值，是否存在，这里为True

print(not True )            # not, “非”运算， 取反

range(x)功能是新建一个表。这个表的元素都是整数，从0开始，下一个元素比前一个大1， 直到函数中所写的上限 （不包括该上限本身），还可以以步进方式创建表。

Python的函数允许不返回值，也就是不用return。默认返回的是None，函数可以返回多个值，逗号隔开，实际返回的是一个tuple元组对象

**参数传递**：对于基本数据类型的变量，变量传递给函数后，函数会在内存中复制一个新的变量，从而不影响原来的变量（我们称此为值传递）。但是对于复杂对象（如三种序列对象）来说，传递给函数的是一个指针，指针指向对象在内存中的位置，在函数中对对象的操作将在原有内存中进行，从而影响原有变量。 （我们称此为指针传递）

定义类的函数时，方法的第一个参数必须是self，无论是否用到，self只限内部使用，不用传参。

子类继承父类的所有属性，包括变量属性和方法属性。

Python有一些特殊方法。Python会特殊的对待它们。特殊方法的特点是名字前后有两个下划线，如类的构造方法。

**python对象属性和类属性，访问修改及内存模型中的大坑需要注意，和其他语言不太一样。**

dir()用来查询一个类或者对象所有属性。

运算符是特殊方法，在类内部定义了该运算符的方法。

---

## Python进阶

词典的元素没有顺序。你不能通过下标引用元素。词典是通过键来引用。

在词典中增添一个新元素的方法：引用一个新的键，并赋予它对应的值。而表list添加元素是append，而元组tuple不能增删改元素。

删除词典元素用del，如del(dict['key1'])

文本文件读写：open(filename,mode)构建文件对象,然后调用file对象操作文件读写。

在Python中，一个.py文件就构成一个模块。通过模块，你可以调用其它文件中的程序。

可以将功能相似的模块放在同一个文件夹中，构成一个模块包。该文件夹中必须包含一个\_\_init\_\_.py的文件，提醒Python，该文件夹为一个模块包。__init__.py可以是一个空文件。

python的函数调用参数传递有几种方式：参数列表的参数位置传递，关键字传递（形参名=实参），也可以混用，但是根据位置传递的参数要正确得放在参数列表前面。

包裹：变长参数，python的变长参数定义时用*name的方式告诉函数这是变长参数，包裹成一个tuple元组。若需要变长的键值对作为参数，即包裹的参数是字典，需要以**name的方式通知函数。

解包裹：在调用函数的时候，需要传入的是固定的几个参数，但是现在数据是一个元组或者字典，则可以直接用解包的方式将该集合元素打散分别作为对应参数传入函数。方式同包裹一样，元组用*，字典用**。

**参数列表顺序**：先位置，再关键字，再包裹位置，再包裹关键字。

enumerate()函数，可以在每次循环中同时得到下标和元素。range函数只能得到元素值，不会得到index。

如果你多个等长的序列，然后想要每次循环时从各个序列分别取出一个元素，可以利用zip()方便地实现，就是依次取矩阵的第一列、第二列……，取出的元素组成一个元组。

循环对象和for循环调用之间还有一个中间层，就是要将循环对象转换成迭代器(iterator)。循环只能针对可循环的对象实施，他们都有next方法

**生成器**：generator，用于自定义循环对象。

理解生成器表达式(Generator Expression):

```python
G = (x for x in range(4))
```

表推导：list comprehension，快速生成表。与生成器类似，利用的是循环对象。

```python
L = [x**2 for x in range(10)] # L=[0,1,4,9,16,25,36,49,64,81]
```

```python
xl = [1,3,5]
yl = [9,12,13]
L  = [ x**2 for (x,y) in zip(xl,yl) if y > 10]
# L = [9,25]
```

函数也是一个对象，具有属性，可用dir()查询。

函数可以作为一个对象，进行参数传递。python函数参数，数值类型是值传递，其他都是引用传递。

lambda生成一个函数对象

```python
func = lambda x,y: x + y # 这里lambda语句生成一个函数对象，引用赋值给func
```

map(func,collection)函数，作用是将函数func依次作用于collection的每个元素，这个函数func可以是lambda语句生成的函数对象。map返回的是一个结果集的循环对象，可用list()函数将其转为list。

```python
re = map((lambda x,y: x+y),[1,2,3],[6,7,9]) 
# map()将每次从两个表中分别取出一个元素，带入lambda所定义的函数
```

filter(func,[10,56,101,500])，filter函数的第一个参数也是一个函数对象。它也是将作为参数的函数对象作用于多个元素。如果函数对象返回的是True，则该次的元素被储存于返回的表中，filter返回的不是表，而是循环对象（同样需要list函数转换）。

```python
def func(a):
    if a > 100:
        return True
    else:
        return False
# 返回的循环对象装换为list为[101,500]，不是Ture False之类的func的返回值
print(filter(func,[10,56,101,500]))
```

reduce函数的第一个参数也是函数，但有一个要求，就是这个函数自身能接收两个参数。reduce可以累进地将函数作用于各个参数。

```python
reduce((lambda x,y: x+y),[1,2,5,7,9]) 
# 相当于(((1+2)+5)+7)+9
# 使用reduce需要导入 functools包
```

异常代码块的执行：try->异常->except->finally，try->无异常->else->finally。也可手动抛出异常，raise 异常对象

对象存储模型：列表可以通过引用其元素，改变对象自身(in-place change)。这种对象类型，称为可变数据对象(mutable object)，词典也是这样的数据类型。而像之前的数字和字符串，不能改变对象本身，只能改变引用的指向，称为不可变数据对象(immutable object)。我们之前学的元组(tuple)，尽管可以调用引用元素，但不可以赋值，因此不能改变对象自身，所以也算是immutable object.

```python
def f(x):
    x[0] = 100
    print(x)

a = [1,2,3]
f(a)
print(a)
# 输出[100,2,3]
# 这里a和x指向同一个对象，但是list对象是可变数据对象，通过引用元素改变了x[0]的值
```

---

## Python深入

Python的运算符是通过调用对象的特殊方法实现的。如+：

```python
'abc' + 'xyz'               # 连接字符串
'abc'.__add__('xyz')        # 实际执行这个特殊方法
```

许多内置函数也都是调用对象的特殊方法，如：

```python
len([1,2,3])      # 返回表中元素的总数
[1,2,3].__len__() # 实际调用
```

在Python中，函数也是一种对象。实际上，任何一个有\_\_call\_\_()特殊方法的对象都被当作是函数。

对于内置的对象来说(比如整数、表、字符串等)，它们所需要的特殊方法都已经在Python中准备好了。而用户自己定义的对象也可以通过增加特殊方法，来实现自定义的语法。

任何定义了\_\_enter\_\_()和\_\_exit\_\_()方法的对象都可以用于上下文管理器。使用上下文管理器以 (with...as...)代码块的形式

```python
with open("new.txt", "w") as f:
    print(f.closed)
    f.write("Hello World!")
print(f.closed) # 到这里，在上面的缩进代码块中f已经调用了它的__exit__函数关闭了
# __exit__(self,exc_type,exc_value,traceback):后是三个参数用于异常处理
```

Python一切皆对象(object)，每个对象都可能有多个属性(attribute)。Python的属性有一套统一的管理方案。

可以利用__class__属性找到对象的类，然后调用类的__base__属性来查询父类

property特性使用内置函数property()来创建。property()最多可以加载四个参数。前三个参数为函数，分别用于处理查询特性、修改特性、删除特性。最后一个参数为特性的文档，可以为一个字符串，起说明作用。



**闭包**

函数是一个对象，所以可以作为某个函数的返回结果。如果定义该函数时，函数内部调用了外部的变量，那么称这个外部变量为这个函数的环境变量，在函数作为返回值返回的时候，这个依赖的环境变量也是默认携带返回的。一个函数和它的环境变量合在一起，构成一个***闭包（closure）***。故闭包就是一个包含环境变量的函数对象，环境变量的值存在这个函数对象的\_\_closure\_\_属性中。闭包可以减少参数，有利于并行计算环境下的编程。

```python
def line_conf(a, b):
    def line(x):
        return a*x + b
    return line

line1 = line_conf(1, 1)
line2 = line_conf(4, 5)
print(line1(5), line2(5))
```



**装饰器**

decorator，类似于装饰模式，或者切面编程，可以用来增强方法或者类。利用的实际是函数对象这个概念。定义装饰器的时候，实际就是在装饰器对象内部定义一个新的函数，再作为返回值返回，调用的时候，直接用@装饰器作用于需要被装饰的函数定义之前。

```python
# 可以对装饰器再加一层装饰，并多添加几个参数，实际上是一种闭包方式
def decorator(F):
    def new_F(a, b):
        print("input", a, b)
        return F(a, b)
    return new_F

# get square sum
@decorator
def square_sum(a, b):
    return a**2 + b**2

# get square diff
@decorator
def square_diff(a, b):
    return a**2 - b**2

print(square_sum(3, 4))
print(square_diff(3, 4))
```



```python
# 类的装饰，为Bird类增加属性total_display,用以增强display方法，记录其调用次数
def decorator(aClass):
    class newClass:
        def __init__(self, age):
            self.total_display   = 0
            self.wrapped         = aClass(age)
        def display(self):
            self.total_display += 1
            print("total display", self.total_display)
            self.wrapped.display()
    return newClass

@decorator
class Bird:
    def __init__(self, age):
        self.age = age
    def display(self):
        print("My age is",self.age)

eagleLord = Bird(5)
for i in range(3):
    eagleLord.display()
```



**内存**

is用于判断两个引用（变量名）所指的对象是否相同（python缓存小的数字和字符，所以is判定为True）。

sys包中的getrefcount()，来查看某个对象的引用计数。但是这个函数一调用，参数本身就会增加一个对象的引用，故该函数返回值会比预期多1.

可以使用del关键字删除某个引用。

手动回收垃圾：gc.collect()，需要导入gc模块

引用计数的GC方式，留意“孤立环问题”

---

## Python补充

使用\_\_name\_\_做单元测试

多种import语法

```python
import TestLib as t
print(t.lib_func(120))

from TestLib import *
print(lib_func(120))

from TestLib import lib_func  #只引用部分对象，减小内存开销
print(lib_func(120))
```



查询对象所属的类和类名称

```python
a = [1, 2, 3]
print a.__class__
print a.__class__.__name__
```



使用中文\#coding=utf8 ，或者 \#-*- coding: UTF-8 -*-



**脚本与命令行结合**

可以使用下面方法运行一个Python脚本，在脚本运行结束后，直接进入Python命令行。这样做的好处是脚本的对象不会被清空，可以通过命令行直接调用。

```bash
$python -i script.py
```



**内置函数**

基本数据类型 type()

反过头来看看 dir() help() len()

词典 len()

文本文件的输入输出 open()

循环设计 range() enumerate() zip()

循环对象 iter()

函数对象 map() filter() reduce()

**在Python中，下列对象都相当于False： [], (), {}, 0, None, 0.0, ''**

---

## Python标准库



**分类** 

***python增强***：文字处理（正则、string）、数据对象（array、queue）、日期时间（time、datetime）、数学运算（random、math）、存储（pickle对象持久化、数据库sqlite3）

***系统互动***：python运行控制（sys包，解释器自己）、操作系统（os包）、线程与进程（threading、multiprocessing）

***网络***：socket、asyncore



**正则，re**

正则表达式(regular expression)主要功能是从字符串(string)中通过特定的模式(pattern)，搜索想要找到的内容。

```python
m = re.search(pattern, string)  # 搜索整个字符串，直到发现符合的子字符串。
m = re.match(pattern, string)   # 从头开始检查字符串是否符合正则表达式。必须从字符串的第一个字符开始就相符。

# 在string中利用正则变换pattern进行搜索，对于搜索到的字符串，用另一字符串replacement替换。返回替换后的字符串。
str = re.sub(pattern, replacement, string) 

re.split()    # 根据正则表达式分割字符串， 将分割后的所有子字符串放在一个表(list)中返回
re.findall()  # 根据正则表达式搜索字符串，将所有符合的子字符串放在一给表(list)中返回
```

正则的书写：看正则文档

^         字符串的起始位置			$         字符串的结尾位置



**时间日期，time、datetime**

```python
import time
# 时间戳
print(time.time())   # wall clock time, unit: second
# 用以测试程序性能，cpu空闲该时间不计时
print(time.clock())  # processor clock time, unit: second

time.sleep(10) # 可以将程序置于休眠状态，定时休眠10秒
```

datetime可以理解为date和time两个组成部分。date是指年月日构成的日期(相当于日历)，time是指时分秒微秒构成的一天24小时中的具体时间(相当于手表)。你可以将这两个分开管理(datetime.date类，datetime.time类)，也可以将两者合在一起(datetime.datetime类)。

```python
# 时间和字符串的转换
from datetime import datetime
format = "output-%Y-%m-%d-%H%M%S.txt" 
str    = "output-1997-12-23-030000.txt" 
t      = datetime.strptime(str, format)
# 反过来，我们也可以调用datetime对象的strftime()方法，来将datetime对象转换为特定格式的字符串
```



**路径与文件，os.path、glob**

```python
import os.path
path = '/home/vamei/doc/file.txt'

print(os.path.basename(path))    # 查询路径中包含的文件名
print(os.path.dirname(path))     # 查询路径中包含的目录

info = os.path.split(path)       # 将路径分割成文件名和目录两个部分，放在一个表中返回
path2 = os.path.join('/', 'home', 'vamei', 'doc', 'file1.txt')  # 使用目录名和文件名构成一个路径字符串

p_list = [path, path2]
print(os.path.commonprefix(p_list))    # 查询多个路径的共同部分

os.path.normpath(path)   # 去除路径path中的冗余。比如'/home/vamei/../.'被转化为'/home'

#-----------------------------------------------------------------------

# os.path还可以查询文件信息
import os.path 
path = '/home/vamei/doc/file.txt'

print(os.path.exists(path))    # 查询文件是否存在

print(os.path.getsize(path))   # 查询文件大小
print(os.path.getatime(path))  # 查询文件上一次读取的时间
print(os.path.getmtime(path))  # 查询文件上一次修改的时间

print(os.path.isfile(path))    # 路径是否指向常规文件
print(os.path.isdir(path))     # 路径是否指向目录文件
```

glob包最常用的方法只有一个, glob.glob(Filename Pattern Expression)，列出所有符合该表达式的文件，类似于命令 ls



**文件管理，shutil**

os包包括各种各样的函数，以实现操作系统的许多功能

```python
from os import *
mkdir(path)
# 创建新目录，path为一个字符串，表示新目录的路径。相当于$mkdir命令

rmdir(path)
#删除空的目录，path为一个字符串，表示想要删除的目录的路径。相当于$rmdir命令

listdir(path)
#返回目录中所有文件。相当于$ls命令。

remove(path)
# 删除path指向的文件。

rename(src, dst)
# 重命名文件，src和dst为两个路径，分别表示重命名之前和之后的路径。 

chmod(path, mode)
# 改变path指向的文件的权限。相当于$chmod命令。

chown(path, uid, gid)
# 改变path所指向文件的拥有者和拥有组。相当于$chown命令。

stat(path)
# 查看path所指向文件的附加信息，相当于$ls -l命令。

symlink(src, dst)
# 为文件dst创建软链接，src为软链接文件的路径。相当于$ln -s命令。

getcwd()
# 查询当前工作路径 (cwd, current working directory)，相当于$pwd命令。
```

shutil包

copy(src, dst) # 复制文件，从src到dst。相当于$cp命令。

move(src, dst) # 移动文件，从src到dst。相当于$mv命令。



**存储对象，pickle**

```python
# 序列化到对象到磁盘文件
import pickle

# define class
class Bird(object):
    have_feather = True
    way_of_reproduction  = 'egg'

summer       = Bird()                 # construct an object
picklestring = pickle.dumps(summer)   # serialize object，文本流字符串

# 直接dump到文件中
fn           = 'a.pkl'
with open(fn, 'w') as f:                     # open file with write-mode
    picklestring = pickle.dump(summer, f)   # serialize and save object
    
# ---------------------------------------------------------------------
# 从文件中反序列化对象
import pickle

# define the class before unpickle，这里不像java有类的字节码，这里需要重新一份一样的类定义给解释器知道该怎么反序列化，jvm可以直接通过class的字节码知道
class Bird(object):
    have_feather = True
    way_of_reproduction  = 'egg'

fn     = 'a.pkl'
with open(fn, 'r') as f:
    summer = pickle.load(f)   # read file and build object
```

cPickle包的功能和用法与pickle包几乎完全相同 (其存在差别的地方实际上很少用到)，不同在于cPickle是基于c语言编写的，速度是pickle包的1000倍。



**子进程，subprocess**

subprocess包主要功能是执行外部的命令和程序，类似与shell。通过标准库中的subprocess包来fork一个子进程，并运行一个外部的程序，另外subprocess还提供了一些管理标准流(standard stream)和管道(pipe)的工具，从而在进程间使用文本通信。



***多线程***

我们在函数中使用global来声明变量为全局变量（定义函数的时候引用外部变量）

threading.Thread对象来代表线程，用threading.Lock对象来代表一个互斥锁 (mutex)。

```python
import threading
import time
import os

# This function could be any function to do other chores.
def doChore():
    time.sleep(0.5)

# Function for each thread
def booth(tid):
    global i
    global lock
    while True:
        lock.acquire()                # Lock; or wait if other thread is holding the lock
        if i != 0:
            i = i - 1                 # Sell tickets
            print(tid,':now left:',i) # Tickets left
            doChore()                 # Other critical operations
        else:
            print("Thread_id",tid," No more tickets")
            os._exit(0)              # Exit the whole process immediately
        lock.release()               # Unblock
        doChore()                    # Non-critical operations

# Start of the main function
i    = 100                           # Available ticket number 
lock = threading.Lock()              # Lock (i.e., mutex)

# Start 10 threads
for k in range(10):
    new_thread = threading.Thread(target=booth,args=(k,))   # Set up thread; target: the callable (function) to be run, args: the argument for the callable 
    new_thread.start()                                      # run the thread
```

面向对象式多线程

```python
import threading
import time
import os

# This function could be any function to do other chores.
def doChore():
    time.sleep(0.5)

# Function for each thread
class BoothThread(threading.Thread):
    def __init__(self, tid, monitor):
        self.tid          = tid
        self.monitor = monitor
        threading.Thread.__init__(self)
    def run(self):
        while True:
            monitor['lock'].acquire()                          # Lock; or wait if other thread is holding the lock
            if monitor['tick'] != 0:
                monitor['tick'] = monitor['tick'] - 1          # Sell tickets
                print(self.tid,':now left:',monitor['tick'])   # Tickets left
                doChore()                                      # Other critical operations
            else:
                print("Thread_id",self.tid," No more tickets")
                os._exit(0)                                    # Exit the whole process immediately
            monitor['lock'].release()                          # Unblock
            doChore()                                          # Non-critical operations

# Start of the main function
monitor = {'tick':100, 'lock':threading.Lock()}

# Start 10 threads
for k in range(10):
    new_thread = BoothThread(k, monitor)
    new_thread.start()
```

定义了一个类BoothThread, 这个类继承自thread.Threading类。然后我们把上面的booth()所进行的操作统统放入到BoothThread类的run()方法中。注意，我们没有使用全局变量声明global，而是使用了一个[词典](http://www.cnblogs.com/vamei/archive/2012/06/06/2537436.html)monitor存放全局变量，然后把词典作为参数传递给线程函数。由于词典是可变数据对象，所以当它被传递给函数的时候，函数所使用的依然是同一个对象，相当于被多个线程所共享。



**数学与随机**

```python
# 从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。
random.choice(seq)  
random.sample(seq,k) # 从序列中随机挑选k个元素
random.shuffle(seq)  # 将序列的所有元素随机排序

random.random()          # 随机生成下一个实数，它在[0,1)范围内。
random.uniform(a,b)      # 随机生成下一个实数，它在[a,b]范围内。

random.gauss(mu,sigma)    # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数。
random.expovariate(lambd) # 随机生成符合指数分布的随机数，lambd为指数分布的参数。
```



**循环器**

在 for i in iterator 结构中，循环器每次返回的对象将赋予给i，直到循环结束。使用iter()内置函数，我们可以将诸如表、字典等容器变为循环器。

标准库中的itertools包提供了更加灵活的生成循环器的工具。

```python
def height_class(h):
    if h > 180:
        return "tall"
    elif h < 160:
        return "short"
    else:
        return "middle"

friends = [191, 158, 159, 165, 170, 177, 181, 182, 190]

friends = sorted(friends, key = height_class)
# 将friends按照height_class函数进行分组，并返回分组标号和元素循环器对象
for m, n in groupby(friends, key = height_class):
    print(m)
    print(list(n))
```



**数据库，sqlite3**

Python自带一个轻量级的关系型数据库SQLite，Python标准库中的sqlite3提供该数据库的接口。

```python
# 创建数据库
import sqlite3

# test.db is a file in the working directory.
conn = sqlite3.connect("test.db")

c = conn.cursor()

# create tables
c.execute('''CREATE TABLE category
      (id int primary key, sort int, name text)''')
c.execute('''CREATE TABLE book
      (id int primary key, 
       sort int, 
       name text, 
       price real, 
       category int,
       FOREIGN KEY (category) REFERENCES category(id))''')

# save the changes
conn.commit()

# close the connection with the database
conn.close()
```

```python
# 插入数据
import sqlite3

conn = sqlite3.connect("test.db")
c    = conn.cursor()

books = [(1, 1, 'Cook Recipe', 3.12, 1),
            (2, 3, 'Python Intro', 17.5, 2),
            (3, 2, 'OS Intro', 13.6, 2),
           ]

# execute "INSERT" 
c.execute("INSERT INTO category VALUES (1, 1, 'kitchen')")

# using the placeholder
c.execute("INSERT INTO category VALUES (?, ?, ?)", [(2, 2, 'computer')])

# execute multiple commands
c.executemany('INSERT INTO book VALUES (?, ?, ?, ?, ?)', books)

conn.commit()
conn.close()
```

```PYTHON
# 查询
import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()

# retrieve one record
c.execute('SELECT name FROM category ORDER BY sort')
print(c.fetchone())
print(c.fetchone())

# retrieve all records as a list
c.execute('SELECT * FROM book WHERE book.category=1')
print(c.fetchall())

# iterate through the records
for row in c.execute('SELECT name, price FROM book ORDER BY sort'):
    print(row)
```

```python
# 更新与删除
import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()

# retrieve one record
c.execute('SELECT name FROM category ORDER BY sort')
print(c.fetchone())
print(c.fetchone())

# retrieve all records as a list
c.execute('SELECT * FROM book WHERE book.category=1')
print(c.fetchall())

# iterate through the records
for row in c.execute('SELECT name, price FROM book ORDER BY sort'):
    print(row)
```

```python
# 删除数据表
c.execute('DROP TABLE book')
```

---

## Python面向对象

**继承、多继承**

Python支持多继承，但是不同类型的类多继承时方法的搜索不同。经典类就是在类定义时没有显示继承Object的类。推荐使用新式类，某类在定义时显示继承object，那么该类的继承树下的所有子类都是新式类。

![img](http://images0.cnblogs.com/blog2015/425762/201508/272341313127410.jpg)

![img](http://images0.cnblogs.com/blog2015/425762/201508/272341553282314.jpg)

- 当类是经典类时，多继承情况下，会按照深度优先方式查找
- 当类是新式类时，多继承情况下，会按照广度优先方式查找

![img](http://images0.cnblogs.com/blog2015/425762/201508/272315068126604.jpg)



类的成员可以分为三大类：字段、方法和属性

![img](http://images2015.cnblogs.com/blog/425762/201509/425762-20150916222236164-249943282.png)

所有成员中，只有普通字段的内容保存对象中，即：根据此类创建了多少对象，在内存中就有多少个普通字段。而其他的成员，则都是保存在类中，即：无论对象的多少，在内存中只创建一份。

每种类的成员都有两种形式：

- 公有成员，在任何地方都能访问
- 私有成员，只有在类的内部才能方法

**私有成员和公有成员的定义不同**：私有成员命名时，前两个字符是下划线（特殊成员除外，例如：\_\_init\_\_、\_\_call\_\_、\_\_dict\_\_等）。

特殊成员：

```python
# __doc__: 类的描述信息
# __module__ 表示当前操作的对象在那个模块
# __class__ 表示当前操作的对象的类是什么
# __init__ 构造方法，通过类创建对象时，自动触发执行
# __del__ 析构方法，当对象在内存中被释放时，自动触发执行。
# __call__ 对象后面加括号，触发执行。
# __dict__ 类或对象中的所有成员(注意成员的内存模型，对应去寻找)
# __str__ 如果一个类中定义了__str__方法，那么在打印 对象 时，默认输出该方法的返回值。相当于toString
# __getitem__、__setitem__、__delitem__用于索引操作，如字典。以上分别表示获取、设置、删除数据
# __getslice__、__setslice__、__delslice__该三个方法用于分片操作
# __iter__ 用于迭代器，让该类对象可以直接用于循环迭代
```



**字段**

字段包括：普通字段和静态字段，他们在定义和使用中有所区别，而最本质的区别是内存中保存的位置不同，

- 普通字段属于**对象**
- 静态字段属于**类**

```python
class Province:

    # 静态字段
    country ＝ '中国'

    def __init__(self, name):

        # 普通字段
        self.name = name


# 直接访问普通字段
obj = Province('河北省')
print obj.name

# 直接访问静态字段
Province.country
```

![img](http://images2015.cnblogs.com/blog/425762/201509/425762-20150907094454965-329821364.jpg)



**方法**

方法包括：普通方法、静态方法和类方法，三种方法在**内存中都归属于类**，区别在于调用方式不同。

- 普通方法：由**对象**调用；至少一个**self**参数；执行普通方法时，自动将调用该方法的**对象**赋值给**self**；
- 类方法：由**类**调用； 至少一个**cls**参数；执行类方法时，自动将调用该方法的**类**复制给**cls**；
- 静态方法：由**类**调用；无默认参数；

```python
class Foo:

    def __init__(self, name):
        self.name = name

    def ord_func(self):
        """ 定义普通方法，至少有一个self参数 """

        # print self.name
        print '普通方法'

    @classmethod
    def class_func(cls):
        """ 定义类方法，至少有一个cls参数 """

        print '类方法'

    @staticmethod
    def static_func():
        """ 定义静态方法 ，无默认参数"""

        print '静态方法'


# 调用普通方法
f = Foo()
f.ord_func()

# 调用类方法
Foo.class_func()

# 调用静态方法
Foo.static_func()
```

![img](http://images2015.cnblogs.com/blog/425762/201510/425762-20151012145051507-726073921.jpg)



**属性**

Python中的属性其实是**普通方法**的变种。

属性的定义有两种方式：

- 装饰器 即：在方法上应用装饰器
- 静态字段 即：在类中定义值为property对象的静态字段

一、装饰器方式

```python
# 经典类，具有一种@property装饰器
class Goods:

    @property
    def price(self):
        return "wupeiqi"
# ############### 调用 ###############
obj = Goods()
result = obj.price  # 自动执行 @property 修饰的 price 方法，并获取方法的返回值
```

```python
# 新式类，具有三种@property装饰器
class Goods(object):

    @property # 属性定义，同时相当于get方法
    def price(self):
        print '@property'

    @price.setter # 相当于set方法
    def price(self, value):
        print '@price.setter'

    @price.deleter # 属性的delete方法
    def price(self):
        print '@price.deleter'

# ############### 调用 ###############
obj = Goods()

obj.price          # 自动执行 @property 修饰的 price 方法，并获取方法的返回值

obj.price = 123    # 自动执行 @price.setter 修饰的 price 方法，并将  123 赋值给方法的参数

del obj.price      # 自动执行 @price.deleter 修饰的 price 方法
```



二、静态字段方式，创建值为property对象的静态字段

```python
class Foo:

    def get_bar(self):
        return 'wupeiqi'

    BAR = property(get_bar)

obj = Foo()
reuslt = obj.BAR        # 自动调用get_bar方法，并获取方法的返回值
print reuslt
```

```python
'''
	property的构造方法中有个四个参数：
    第一个参数是方法名，调用 对象.属性 时自动触发执行方法
    第二个参数是方法名，调用 对象.属性 ＝ XXX 时自动触发执行方法
    第三个参数是方法名，调用 del 对象.属性 时自动触发执行方法
    第四个参数是字符串，调用 对象.属性.__doc__ ，此参数是该属性的描述信息

'''
class Foo：

    def get_bar(self):
        return 'wupeiqi'

    # *必须两个参数
    def set_bar(self, value): 
        return return 'set value' + value

    def del_bar(self):
        return 'wupeiqi'

    BAR ＝ property(get_bar, set_bar, del_bar, 'description...')

obj = Foo()

obj.BAR              # 自动调用第一个参数中定义的方法：get_bar
obj.BAR = "alex"     # 自动调用第二个参数中定义的方法：set_bar方法，并将“alex”当作参数传入
del Foo.BAR          # 自动调用第三个参数中定义的方法：del_bar方法
obj.BAE.__doc__      # 自动获取第四个参数中设置的值：description...
```




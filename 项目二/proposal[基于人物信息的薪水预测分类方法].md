[TOC]

---

# 基于人物信息的薪水预测分类方法

​    

## Project Definition

本文通过对获得的大量人物信息（如年龄、姓名、学历、婚姻、薪水等）做挖掘分析，得出一个具体的模型用以描述个人信息（除薪水属性）与其薪水的关系，通过这个模型最终达到的效果是可以通过人物相关信息得出其薪水的预测。根据所给数据集的特征，为简化工作，本文仅对薪水是否大于50K做出预测。

针对上述内容，本文将主要开展以下工作：

- 获取数据集，并完成数据预处理。
- 采用不同的分类方法，在训练样本上执行分类器算法，生成分类模型。
- 在测试样本上执行分类模型，生成预测结果。
- 根据预测结果，对不同的分类方法得出的分类模型性能做对比评估。


   


## Motivation

本文的研究主要有以下两个动因：

- 作为《数据挖掘》课程实践项目，实现课程理论到实践的初步过渡，简单了解数据挖掘在现实生产的流程。
- 本文研究内容同时具有实际的参考意义，如消费网站可根据用户信息预测其经济能力，做出针对性的商品推荐；银行通过用户信息预测其薪水能对该用户做贷款风险评估；招聘网站可根据用户信息推送其合适薪水的岗位。


  

## Background

目前关于分类算法的研究非常多，不论在理论研究还是实际生产中都已经非常成熟。关于分类方法的研究主要集中在以下几类。

- 决策树分类算法（DT，Decision Tree）。决策树方法是利用信息论中的信息增益寻找数据库中具有最大信息量的属性字段，建立决策树的一个结点，再根据该属性字段的不同取值建立树的分支，在每个子分支子集中重复建立树的下层结点和分支的一个过程。构造决策树的具体过程为：首先寻找初始分裂，整个训练集作为产生决策树的集合，训练集每个记录必须是已经分好类的，以决定哪个属性域（Field）作为目前最好的分类指标。一般的做法是穷尽所有的属性域，对每个属性域分裂的好坏做出量化，计算出最好的一个分裂。量化的标准是计算每个分裂的多样性（Diversity）指标。其次，重复第一步，直至每个叶节点内的记录都属于同一类且增长到一棵完整的树。
- 支持向量机（SVMs，Support Vector Machines）。支持向量机（SVM）方法是建立在统计学习理论的VC维和结构风险最小原理基础上的，根据有限的样本信息在模型的复杂性和学习能力之间寻求最佳折衷，以期获得最好的推广能力。SVM是从线性可分情况下的最优分类面发展而来的，使分类间隔最大实际上就是对推广能力的控制，这是SVM的核心思想之一。由于统计学习理论和支持向量机建立了一套较好的在小样本下机器学习的理论框架和通用方法，既有严格的理论基础，又能较好地解决小样本、高维和局部极小点等实际问题，因此成为继神经网络之后的又一个研究方向。但是，处理大规模数据集时，SVM速度慢，往往需要较长的训练时间。而且，SVM方法需要计算和存储核函数矩阵，当样本数目较大时，需要很大的内存。其次，SVM在二次型寻优过程中要进行大量的矩阵运算，多数情况下，寻优算法是占用算法时间的主要部分。
- 基于关联规则的分类（CBA，Classification Based on Association Rule）。挖掘关联规则就是发现大量数据中项集之间有趣的关联或相关联的过程。关联规则挖掘用于分类问题取得了很好的效果。主要有Apriori算法和LIG（large items generation）算法。
- K-近邻算法（kNN，k-Nearest Neighbors）。一种基于实例的非参数分类方法。KNN分类算法搜索样本空间，计算未知类别向量与样本集中每个向量的相似度，在样本集中找出K个最相似的文本向量，分类结果为相似样本中最多的一类。
- 贝叶斯分类算法。属于统计学分类算法，利用概率统计知识进行分类的算法，目前比较受关注的是朴素贝叶斯分类算法（NB，Naive Bayes）。
- 逻辑回归（LR，Logistic Regression）。面对一个回归或者分类问题，建立代价函数，然后通过优化方法迭代求解出最优的模型参数，然后测试验证我们这个求解的模型的好坏。Logistic回归虽然名字里带“回归”，但是它实际上是一种分类方法，主要用于两分类问题。


基于现有的算法研究成果，本文不再着重算法的理论研究，而主要将精力放在各分类算法的实现以及算法评估和比较上。

  

## Data Set

[http://archive.ics.uci.edu/ml/datasets/Adult](http://archive.ics.uci.edu/ml/datasets/Adult)

- 数据量：48842
- 数据集特征：多变量
- 领域：社交
- 属性特征：多类型
- 属性数量：14
- 是否存在数据缺失：是

该数据集数据量适中，可适用多种分类方法进行分类。属性数目较小，针对该数据集的处理复杂度不会太高，且能应对数据缺失的情况，该数据集也是目前数据挖掘研究中比较常用的数据集。

  

## Software

- python + pycharm + scikit-learn：本文代码实现部分采用python实现，IDE选用pycharm，关于分类算法会采用开源的库scikit-learn。
- 算法：本文将选用background中提及的多种算法中的部分算法，参考开源库中的实现代码，采用python语言实现。


  

## Evaluation Method

有两种方法可以用于对分类器的错误率进行评估，它们都假定待预测记录和训练集取自同样的样本分布。

- 保留方法(Holdout)：记录集中的一部分（通常是2/3）作为训练集，保留剩余的部分用作测试集。生成器使用2/3 的数据来构造分类器，然后使用这个分类器来对测试集进行分类，得出的错误率就是评估错误率。虽然这种方法速度快，但由于仅使用2/3 的数据来构造分类器，因此它没有充分利用所有的数据来进行学习。如果使用所有的数据，那么可能构造出更精确的分类器。
- 交叉纠错方法(Cross Validation)：数据集被分成k个没有交叉数据的子集，所有子集的大小大致相同。生成器训练和测试共k次；每一次，生成器使用去除一个子集的剩余数据作为训练集，然后在被去除的子集上进行测试。把所有得到的错误率的平均值作为评估错误率。交叉纠错法可以被重复多次(t)，对于一个t次k 分的交叉纠错法，k *t 个分类器被构造并被评估，这意味着交叉纠错法的时间是分类器构造时间的k *t 倍。增加重复的次数意味着运行时间的增长和错误率评估的改善。可以对k 的值进行调整，将它减少到3 或5，这样可以缩短运行时间。然而，减小训练集有可能使评估产生更大的偏差。通常Holdout评估方法被用在最初试验性的场合，或者多于5000条记录的数据集；交叉纠错法被用于建立最终的分类器，或者很小的数据集。


本文将采用保留方法对分类器做错误率评估，将整个数据集分为32561条记录组成的训练集和16281条记录组成的测试集。若时间充足，将增加交叉纠错方法评估。

​        


## Milestones And Plan

- 10月13日前，完成proposal，对整个项目有个粗略的计划，该计划可能在后期的学习中因认知的提升不断修改。此期间需要完成数据的收集。
- 11月3日前，完成第一篇进度报告，包含以下内容：
  - 数据预处理，将文本数据整理成需要的格式。
  - 完成python语言的学习。
  - 学习一种分类算法的理论，尝试采用python实现，或者至少能应用已有开源库做一些该算法的demo。
- 12月1日前，完成第二篇进度报告，包含下述内容：
  - 继续开展各种分类算法的理论学习。
  - 尝试python实现这些算法。
  - 开展实验，采用这些算法对数据进行处理。
- 结课前，完成最终的项目报告，包含以下内容：
  - 用图表对第二进度的实验结果进行汇总比较，评估算法。
  - 整理代码与文献资料。
  - 学习总结或后续进一步的研究计划。


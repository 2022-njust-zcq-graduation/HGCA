### HGCA

Heterogeneous Graph Contrastive Learning with Adaptive Augmentation（异质图的自适应增强对比表示学习方法）

![](https://static01.imgkr.com/temp/d903e886cb594f0f8b75710dfd2f8ef6.png)

### 概述

我们在这里用PyTorch提供了HGCA的一个实现，以及在IMDB数据集和ACM数据集的示例。文件组织结构如下：

- ```data/```包含IMDB数据集文件和ACM数据集文件
- ```layers/```包含图卷积层（```gcn.py```)、语义级别的注意力层（```attentionLayer.py```)、组合多个图卷积层和一个语义级别注意力层的异质图卷积层（```hgcn.py```)
- ```models/```包含异质图对比表示学习方法模型（```hgca.py```）、下游节点分类任务的逻辑回归分类器（```logreg.py```）
- ```utils/```包含自适应增强方法（```functional.py```）、加载数据和归一化数据以及评分函数（```process.py```）
- ```temp/```保存了IMDB数据集和ACM数据集节点表示学习时最优的模型

最后，```execute.py```是将以上内容组合起来的完整训练过程，会保存训练过程中最优的节点表示模型（```best_hgca.pkl```）和下游分类任务中最优的模型（```best_test.pkl```）以及完整的输出日志（```HGCA.txt```）

```half_execute.py```是部分训练过程，它直接读取```temp/```中保存的节点表示模型进行下游节点分类任务的训练，保存了仅测试过程的输出日志（```temp/HGCA.txt```）
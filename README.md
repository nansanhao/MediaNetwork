# 网络媒体分析 - 实验一网络测度
## 一、 实验目的
1、掌握网络数据表达及其可视化方法。

2、了解典型的网络测度。

## 二、 实验环境
硬件：计算机。

软件：计算机程序语言开发平台，如C、C++、Java、Matlab。本实验选择了python作为开发语言。
## 三、 实验内容及过程
1. 我们采用python作为开发语言，使用networkx和matplotlib.pyplot两个包来描绘图的结构，将数据集的节点和边描绘出来。如下：（黄色的是节点，黑色的是边）
![结果图](https://github.com/nansanhao/MediaNetwork/blob/master/graph_1.png?raw=true)

2. 测度（Contents from ppt）

    1 Centrality

        1.1 degree centrality（度中心性）
        1.2 normalized degree centrality（标准化度中心性）
        1.3 Eigenvector centrality（特征向量中心性）
        1.4 Kartz Centality（Katz中心性）
        1.5 Pagerank（网页排名）
        1.6 Betweenness Centrality（间接中心性）
        1.7 Closeness Centrality（紧密中心性）
        1.8 Group Centality

    2 Transitivity and Reciprocity

        2.1 Transitivity
            2.1.1 global
        2.2 Reciprocity（互易性）

    3 Balance and Status 

    4 Similarity（相似性）

        4.1 Similarity
        4.2 Regular Equivalence


## 文件说明：
    test_1.py是第一问代码
    test_2.py是第二问代码
    data开头的是数据集文件夹


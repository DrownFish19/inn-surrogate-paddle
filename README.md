# 飞桨论文复现挑战赛（第七期）科学计算 (ID:50 inn-surrogate)

[English](./README_en.md) | 简体中文

- [飞桨论文复现挑战赛（第七期）科学计算 (ID:50 inn-surrogate)](#飞桨论文复现挑战赛（第七期）科学计算id-50-inn-surrogate)
  - [1.简介](#1简介)
    - [2D模型](#2d模型)
    - [3D模型](#3d模型)
  - [2.数据集](#2数据集)
    - [数据下载使用](#数据下载使用)
    - [数据集含义](#数据集含义)
    - [数据集对应关系](#数据集对应关系)
  - [3.环境依赖](#3环境依赖)
    - [硬件](#硬件)
    - [框架](#框架)
    - [本地安装](#本地安装)
  - [4.快速开始](#4快速开始)
    - [aistudio](#aistudio)
    - [本地运行](#本地运行)
  - [5.代码结构与详细说明](#5代码结构与详细说明)
  - [6.复现结果](#6复现结果)
    - [原论文2D模型结果](#原论文2d模型结果)
    - [复现2D模型结果](#复现2d模型结果)
    - [原论文3D模型结果](#原论文3d模型结果)
    - [复现3D模型结果](#复现3d模型结果)
  - [7.模型信息](#7模型信息)


## 1.简介

本项目基于paddle框架复现，论文主要亮点如下：
* 作者不是为正向模型开发代理，而是直接训练一个反向代理，将物理系统的输出信息映射到未知的输入分布式参数。
* 建立了一种基于条件可逆神经网络（cINN）的生成模型。
* cINN被训练为作为由偏微分方程控制的物理系统反向替代模型。
* 逆代理模型用于求解具有未知空间相关参数的逆问题。
* 该方法应用于利用有限压力和饱和度数据估计多相流中的非高斯渗透率场。

论文信息：
* Anantha Padmanabha G, Zabaras N. Solving inverse problems using conditional invertible neural networks[J]. Journal of Computational Physics, 2021, 433: 110194. 

参考GitHub地址：
* https://github.com/zabaras/inn-surrogate

项目aistudio地址：
* https://aistudio.baidu.com/aistudio/projectdetail/4756062

### 2D模型
![](https://ai-studio-static-online.cdn.bcebos.com/3d038fb4ba6543c4b346161927a47500455a9bc8703b418eb32f08d9c7611ede)


### 3D模型
![](https://ai-studio-static-online.cdn.bcebos.com/2990489d1e3c4435ac1c35807626effd0889f0d9b402468f8885e9cc729ca8d3)


## 2.数据集

### 数据下载使用
数据集包含2D数据和3D数据，可通过[此处链接](https://zenodo.org/record/4631233#.YFo8N-F7mDI)进行下载。

数据集aistudio地址：
* 2D数据: https://aistudio.baidu.com/aistudio/datasetdetail/166997
* 3D数据: https://aistudio.baidu.com/aistudio/datasetdetail/167243

本项目已经关联以上数据集，可直接在项目data文件夹下找到对应数据集

### 数据集含义
具体数据集组成及含义可参考论文4.1节Identiﬁcation of the permeability ﬁeld of an oil reservoir。

### 数据集对应关系
数据集划分为train，test和sample数据集。train数据集中包含10000样本，但可通过args函数指定训练集数量用于训练。文件名称中包含1pc，3pc，5pc分别对应1%，3%，5%的噪声。


## 3.环境依赖

### 硬件
* 2D模型：gpu memory >= 16GB (v100 16g)
* 3D模型：gpu memory >= 32GB (v100 32g)

### 框架
* paddle >= 2.2.0
* matplotlib
* h5py
* scipy
* scikit-learn

### 本地安装
```bash
conda create -n paddle_env python=3.8
conda install paddlepaddle==2.2.2
conda install scipy h5py matplotlib scikit-learn
```

## 4.快速开始

### aistudio
本项目已经构建notebook（同[X4Science/INFINITY](http://https://github.com/X4Science/INFINITY)中相同代码）用于快速实现。具体代码请参考以下notebook，运行中需修改数据路径和结果路径参数请在`args`中修改。
* 2D模型：main-2D.ipynb (备注：推荐使用v100 32g计算卡)
* 3D模型：main-3d.ipynb (备注：推荐使用A40  40g计算卡)

### 本地运行
* 下载数据集文件
* 从github下载本项目代码，[X4Science/INFINITY](http://https://github.com/X4Science/INFINITY)
* 运行

```bash
cd 2D # or cd 3D
python train.py
```


## 5.代码结构与详细说明

```txt
inn-surrogate-paddle
|-- 2D                                      # 2D模型文件夹
|   |-- args.py                             # 配置
|   |-- models                              # 模型文件夹
|   |   |-- CouplingBlock.py                
|   |   |-- CouplingOneSide.py
|   |   |-- Divide_data_model.py
|   |   |-- Downsample_model.py
|   |   |-- Permute_data_model.py
|   |   |-- Unflat_data_model.py
|   |   |-- conditioning_network.py
|   |   |-- flat_data_model.py
|   |   `-- main_model.py
|   |-- train.py                            # 训练代码
|   `-- utils                               # 工具代码
|       |-- load_data.py
|       |-- plot.py
|       `-- plot_samples.py
|-- 3D                                      # 3D模型文件夹
|   |-- args.py                             # 配置
|   |-- models                              # 模型文件夹
|   |   |-- CouplingBlock_model.py
|   |   |-- CouplingOneSide_model.py
|   |   |-- Divide_data_model.py
|   |   |-- Downsample_model.py
|   |   |-- Permute_data_model.py
|   |   |-- Unflat_data_model.py
|   |   |-- conditioning_network.py
|   |   |-- flat_data_model.py
|   |   `-- main_model.py
|   |-- train.py                            # 训练代码      
|   `-- utils                               # 工具代码
|       |-- error_bars.py
|       |-- load_data.py
|       `-- plot.py
`-- data                                    # 数据集
    |-- 2D_problem_dataset                  # 2D模型数据集
    |   |-- Config_2_sample_obs_1pc.hdf5
    |   |-- Config_2_sample_obs_3pc.hdf5
    |   |-- Config_2_sample_obs_5pc.hdf5
    |   |-- Config_2_test_obs_1pc.hdf5
    |   |-- Config_2_test_obs_3pc.hdf5
    |   |-- Config_2_test_obs_5pc.hdf5
    |   |-- Config_2_train_obs_1pc.hdf5
    |   |-- Config_2_train_obs_3pc.hdf5
    |   `-- Config_2_train_obs_5pc.hdf5
    `-- 3D_problem_dataset                  # 3D模型数据集
        |-- Config_2_sample_obs_1pc_3D.hdf5
        |-- Config_2_sample_obs_3pc_3D.hdf5
        |-- Config_2_sample_obs_5pc_3D.hdf5
        |-- Config_2_test_obs_1pc_3D.hdf5
        |-- Config_2_test_obs_3pc_3D.hdf5
        |-- Config_2_test_obs_5pc_3D.hdf5
        |-- Config_2_train_obs_1pc_3D.hdf5
        |-- Config_2_train_obs_3pc_3D.hdf5
        `-- Config_2_train_obs_5pc_3D.hdf5
```


## 6.复现结果

### 原论文2D模型结果
![](https://ai-studio-static-online.cdn.bcebos.com/09d8e2b5fe5a473b899842845a2ece667c2804a9c88c419caa27719de77ec790)

### 复现2D模型结果
![](https://ai-studio-static-online.cdn.bcebos.com/5128939ec4bf445a84ae0212f9dd7ac1178e0dbcf8a0447cbb12672b6f24ddbc)

### 原论文3D模型结果
![](https://ai-studio-static-online.cdn.bcebos.com/659d5e3a98954b9bafb32f9f4f4b707ebd7370182aa54606933db0e08a22746b)

### 复现3D模型结果
![](https://ai-studio-static-online.cdn.bcebos.com/8d329d7416a040d391b11d6e771a6d522027b75d466041ea836fdbc51b42407b)


## 7.模型信息

| 信息                | 说明| 
| --------          | -------- | 
| 发布者               | 朱卫国 (DrownFish19)    | 
| 发布时间              | 2022.10     | 
| 框架版本              | paddle 2.3.2     | 
| 支持硬件              | GPU、CPU     | 
| 模型下载链接           |  [预训练模型](https://pan.baidu.com/s/14fUU1YU-b-zkkQPkdl_JzQ?pwd=82ij)    | 
| 预训练模型训练时间-2D (A40) （6000，8000，10000样本量）             | 1.2h,1.6h,2h     |
| 预训练模型训练时间-3D (A40) （6000，8000，10000样本量）            | 5.3h,7,0h,8.75h     |
| aistudio              | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4756062)     | 

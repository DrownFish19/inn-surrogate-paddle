# 飞桨论文复现挑战赛（第七期）科学计算 (ID:50 inn-surrogate)

English | [简体中文](./README.md)

- [飞桨论文复现挑战赛（第七期）科学计算 (ID:50 inn-surrogate)](#飞桨论文复现挑战赛（第七期）科学计算id-50-inn-surrogate)
  - [1.Introduction](#1-introduction)
    - [2D model](#2d-model)
    - [3D model](#3d-model)
  - [2.Dataset](#2-dataset)
    - [dataset download](#dataset-download)
    - [meaning](#meaning)
    - [format](#format)
  - [3.Environment](#3-environment)
    - [Hardware](#hardware)
    - [Framework](#framework)
    - [Local Environment](#local-environment)
  - [4.Quick start](#4-quick-start)
    - [aistudio](#aistudio)
    - [Local Running](#local-running)
  - [5.Code structure](#5-code-structure)
  - [6.Reproduced Results](#6-reproduced-results)
    - [The results of the 2D model (origin)](#the-results-of-the-2d-model-origin)
    - [The results of the 2D model (reproduced)](#the-results-of-the-2d-model-reproduced)
    - [The results of the 3D model (origin)](#the-results-of-the-3d-model-origin)
    - [The results of the 3D model (reproduced)](#the-results-of-the-3d-model-reproduced)
  - [7.Model information](#7-model-information)


## 1.Introduction

This project reproduces based on the PaddlePaddle framework. The highlights are summarized as follows
* Rather than developing a surrogate for a forward model, we are training directly an inverse surrogate mapping output information of a physical system to an unknown input distributed parameter.
* A generative model based on conditional invertible neural networks (cINN) is developed.
* The cINN is trained to serve as an inverse surrogate model of physical systems governed by PDEs.
* The inverse surrogate model is used for the solution of inverse problems with unknown spatially-dependent parameters.
* The developed method is applied for the estimation of a non-Gaussian permeability field in multiphase flows using limited pressure and saturation data.

Paper：
* Anantha Padmanabha G, Zabaras N. Solving inverse problems using conditional invertible neural networks[J]. Journal of Computational Physics, 2021, 433: 110194. 

Reference GitHub：
* https://github.com/zabaras/inn-surrogate

The link of aistudio：
* https://aistudio.baidu.com/aistudio/projectdetail/4756062

### 2D model
![](https://ai-studio-static-online.cdn.bcebos.com/3d038fb4ba6543c4b346161927a47500455a9bc8703b418eb32f08d9c7611ede)


### 3D model
![](https://ai-studio-static-online.cdn.bcebos.com/2990489d1e3c4435ac1c35807626effd0889f0d9b402468f8885e9cc729ca8d3)


## 2.Dataset

### dataset download
The dataset includes 2D and 3D data for this project, which can be downloaded from [this link](https://zenodo.org/record/4631233#.YFo8N-F7mDI).

The link of the dataset in aistudio：
* 2D data: https://aistudio.baidu.com/aistudio/datasetdetail/166997
* 3D data: https://aistudio.baidu.com/aistudio/datasetdetail/167243

### meaning
Please refer to Section 4.1 Identiﬁcation of the permeability ﬁeld of an oil reservoir in the paper.

### format
The dataset includes train, test, and sample data, in which the train data contains 10000 samples. In the filename, the 1pc, 3pc, and 5pc indicate observations with 1%, 3%, and 5% independent Gaussian noise, respectively.

## 3.Environment

### Hardware
* 2D model：gpu memory >= 16GB (v100 16g)
* 3D model：gpu memory >= 32GB (v100 32g)

### Framework
* paddle >= 2.2.0
* matplotlib
* h5py
* scipy
* scikit-learn

### Local Environment
```bash
conda create -n paddle_env python=3.8
conda install paddlepaddle==2.2.2
conda install scipy h5py matplotlib scikit-learn
```

## 4.Quick start

### aistudio
The project has built a notebook version for a quick start (the code is from [X4Science/INFINITY](http://https://github.com/X4Science/INFINITY)).
* 2D model：main-2D.ipynb (recommend v100 32g)
* 3D model：main-3d.ipynb (recommend A40  40g)

### Local Running
* Download datasets.
* clone from GitHub，[X4Science/INFINITY](http://https://github.com/X4Science/INFINITY)
* run

```bash
cd 2D # or cd 3D
python train.py
```


## 5.Code structure

```txt
inn-surrogate-paddle
|-- 2D                                      # 2D model folder
|   |-- args.py                             # configuration
|   |-- models                              # model folder
|   |   |-- CouplingBlock.py                
|   |   |-- CouplingOneSide.py
|   |   |-- Divide_data_model.py
|   |   |-- Downsample_model.py
|   |   |-- Permute_data_model.py
|   |   |-- Unflat_data_model.py
|   |   |-- conditioning_network.py
|   |   |-- flat_data_model.py
|   |   `-- main_model.py
|   |-- train.py                            # training code
|   `-- utils                               # util tools code
|       |-- load_data.py
|       |-- plot.py
|       `-- plot_samples.py
|-- 3D                                      # 3D model folder
|   |-- args.py                             # configuration
|   |-- models                              # model folder
|   |   |-- CouplingBlock_model.py
|   |   |-- CouplingOneSide_model.py
|   |   |-- Divide_data_model.py
|   |   |-- Downsample_model.py
|   |   |-- Permute_data_model.py
|   |   |-- Unflat_data_model.py
|   |   |-- conditioning_network.py
|   |   |-- flat_data_model.py
|   |   `-- main_model.py
|   |-- train.py                            # training code     
|   `-- utils                               # util tools code 
|       |-- error_bars.py
|       |-- load_data.py
|       `-- plot.py
`-- data                                    # Dataset
    |-- 2D_problem_dataset                  # 2D model 
    |   |-- Config_2_sample_obs_1pc.hdf5
    |   |-- Config_2_sample_obs_3pc.hdf5
    |   |-- Config_2_sample_obs_5pc.hdf5
    |   |-- Config_2_test_obs_1pc.hdf5
    |   |-- Config_2_test_obs_3pc.hdf5
    |   |-- Config_2_test_obs_5pc.hdf5
    |   |-- Config_2_train_obs_1pc.hdf5
    |   |-- Config_2_train_obs_3pc.hdf5
    |   `-- Config_2_train_obs_5pc.hdf5
    `-- 3D_problem_dataset                  # 3D model
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


## 6.Reproduced Results

### The results of the 2D model (origin)
![](https://ai-studio-static-online.cdn.bcebos.com/09d8e2b5fe5a473b899842845a2ece667c2804a9c88c419caa27719de77ec790)

### The results of the 2D model (reproduced)
![](https://ai-studio-static-online.cdn.bcebos.com/5128939ec4bf445a84ae0212f9dd7ac1178e0dbcf8a0447cbb12672b6f24ddbc)

### The results of the 3D model (origin)
![](https://ai-studio-static-online.cdn.bcebos.com/659d5e3a98954b9bafb32f9f4f4b707ebd7370182aa54606933db0e08a22746b)

### The results of the 3D model (reproduced)
![](https://ai-studio-static-online.cdn.bcebos.com/8d329d7416a040d391b11d6e771a6d522027b75d466041ea836fdbc51b42407b)


## 7.Model information

| information                | description| 
| --------          | -------- | 
| Author              | Weiguo Zhu (DrownFish19)    | 
| Date              | 2022.10     | 
| Framework version              | paddle 2.3.2     | 
| Support hardware              | GPU、CPU     | 
| Download link           |  [Pre-training models](https://pan.baidu.com/s/14fUU1YU-b-zkkQPkdl_JzQ?pwd=82ij)    | 
| Training Time-2D (A40) （6000，8000，10000 samples）             | 1.2h,1.6h,2h     |
| Training Time-3D (A40) （6000，8000，10000 samples）            | 5.3h,7,0h,8.75h     |
| aistudio              | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4756062)     | 
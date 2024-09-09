<h1 align="center">ClusteringSDF: Self-Organized Neural Implicit Surfaces for 3D Decomposition</h1>
  <p align="center">
    <a href="https://sm0kywu.github.io/CV/CV.html">Tianhao Wu</a>
    ·
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    ·
    <a href="https://wuqianyi.top/">Qianyi Wu</a>
    .
    <a href="https://personal.ntu.edu.sg/astjcham/index.html">Tat-Jen Cham</a>

  </p>
  <h3 align="center">ECCV 2024</h3>
  <h3 align="center"><a href="https://arxiv.org/pdf/2403.14619">Paper</a> | <a href="https://sm0kywu.github.io/ClusteringSDF/">Project Page</a></h3>
  <div align="center"></div>
</p>

## Setup

### Installation
This code has been tested on Ubuntu 22.02 with torch 2.0 & CUDA 11.7.

```
cd ClusteringSDF
conda create -y -n clusteringsdf python=3.9
conda activate clusteringsdf
pip install -r requirements.txt
```

### Dataset
The data are obtained same as [Panoptic Lifting](https://github.com/nihalsid/panoptic-lifting). We have resized the images in our paper and modified the camera pose accordingly to meet computing resource constraint, which can be adjusted in the config file. Depth and normal guidance are obtained from [omnidata](https://github.com/EPFL-VILAB/omnidata). You can download the sample Replica data [here](https://drive.google.com/file/d/1_IdZuX0Y130G5w26YskwHqqJgVI4oxEb/view?usp=sharing).


### Training and inference

```
cd code
bash train.sh
bash inference.sh
```


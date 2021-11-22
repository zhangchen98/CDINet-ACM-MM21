<div align=center>
<img src = https://2021.acmmm.org/img/acmmm2021_logo.1f9d3343.png>
</div>

## CDINet

This is a PyTorch implementation of "Cross-modality Discrepant Interaction Network for RGB-D Salient Object Detection" accepted by ACM MM 2021 (poster).

Paper: https://dl.acm.org/doi/pdf/10.1145/3474085.3475364. 

Arxiv version: https://arxiv.org/pdf/2108.01971.pdf

## Network
![image](https://user-images.githubusercontent.com/45169768/129518120-d6ffaf4c-896b-4030-8b48-fc586998b417.png)

## Requirement

Pleasure configure the environment according to the given version:

- python 3.7.10
- pytorch 1.8.0
- cudatoolkit 10.2.89
- torchvision 0.9.0
- tensorboardx 2.3
- opencv-python 4.5.1.48
- numpy 1.20.2



We also provide ".yaml" files for conda environment configuration, you can download it from [[Link]](https://pan.baidu.com/s/1o7yo6_86K1Ey6ZHhpiwRyg), code: 642h, then use `conda env create -f CDINet.yaml` to create a required environment.



## Data Preprocessing

For all depth maps in training and testing datasets, we make a uniform adjustment so that  the foreground have higher value than the background, it is very important. Please follow the tips to download the processed datasets and pre-trained model:

1. Download training data  from [[Link](https://pan.baidu.com/s/1jm-B10GfOinp9G17VsxH_A)], code: 0812.
2. Download testing data from [[Link](https://pan.baidu.com/s/1PncdQcU5jptqYjfwJfBopA)], code: 0812.
3. Download the parameters of whole model from  [[Link](https://pan.baidu.com/s/1VeGAIR30jQoWvZHDVxeTow)], code: 0812.

```python
├── backbone 
├── CDINet.pth
├── CDINet_test.py
├── CDINet_train.py
├── dataset
│   ├── CDINet_test_data
│   └── CDINet_train_data
├── model
├── modules
└── setting
```





## Training and Testing

**Training command**: `python CDINet_train.py --gpu_id xx  --batchsize xx`

You can find the saved models and logs in "./CDINet_cpts".


**Testing command**: `python CDINet_test.py --gpu_id xx` 

You can find the saliency maps in "./saliency_maps".



## Results

1. **Qualitative results**: we provide the saliency maps, you can download them from [[Link](https://pan.baidu.com/s/1yDlwuOgqTKkO3LDXqyfQ2w)], code: 0812.
2. **Quantitative results**: 


|              |  NLPR  |  NJUD  |  DUT   | STEREO |  LFSD  |
| :----------: | :----: | :----: | :----: | :----: | :----: |
| ![](https://latex.codecogs.com/svg.image?F_{max}) | 0.9162 | 0.9215 | 0.9372 | 0.9033 | 0.8746 |
| ![](https://latex.codecogs.com/svg.image?S_{\alpha}) | 0.9273 | 0.9188 | 0.9274 | 0.9055 | 0.8703 |
| ![](https://latex.codecogs.com/svg.image?MAE)   | 0.0240 | 0.9354 | 0.0302 | 0.0410 | 0.0631 |


## Bibtex

```
@inproceedings{Zhang2021CDINet, 
    author = {Zhang, Chen and Cong, Runmin and Lin, Qinwei and Ma, Lin and Li Feng and Zhao, Yao and Kwong, Sam},   
    title = {Cross-modality Discrepant Interaction Network for {RGB-D} Salient Object Detection},     
    booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},     
    year = {2021},
    organization={ACM}
} 
```



## Contact

If you have any questions, please contact Chen Zhang at [chen.zhang@bjtu.edu.cn](mailto:chen.zhang@bjtu.edu.cn) .

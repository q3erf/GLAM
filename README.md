# GLAM
This repo is the unofficial implementation of paper 
["Joint Graph Learning and Matching for Semantic Feature Correspondence"](https://arxiv.org/abs/2109.00240)

## Install
1. create conda environment
```bash
conda create -n GLAM python=3.8 
```
2. conda install pytorch
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Data Preparation

download spair71-K and unzip in /data/downloaded/ 
```html
http://cvlab.postech.ac.kr/research/SPair-71k/
```
## Run
```bash
python3 train.py ./experiments/spair.json
```
## Credits and Citation
Please cite the following paper if you use this model in your research:
```
Liu H, Wang T, Li Y, et al. Joint Graph Learning and Matching for Semantic Feature Correspondence[J]. arXiv preprint arXiv:2109.00240, 2021.
```
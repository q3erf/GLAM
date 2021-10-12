# GLAM
*Joint Graph Learning and Matching for Semantic Feature Correspondence*

# install
1. create conda environment
```bash
conda create -n GLAM python=3.8 
```
2. conda install pytorch
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

# data  preparation

download spair71-K and unzip in /data/downloaded/ 
```html
http://cvlab.postech.ac.kr/research/SPair-71k/
```
# run
```bash
python3 train.py ./experiments/spair.json
```

Liu H, Wang T, Li Y, et al. Joint Graph Learning and Matching for Semantic Feature Correspondence[J]. arXiv preprint arXiv:2109.00240, 2021.
## Install

### install torch

```
conda create -n <your env> python=3.9
# select specific CUDA version
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### pip install

```
pip install -r requirements.txt
pip install monai[itk]
```

## Usage

### config
```
export DATASET_LOCATION=~/add/your/data/here
export EXPERIMENT_LOCATION=~/add/your/exp/result/here
```

### run
```
python train.py -m dataset_folder.dataset.fold=0,1,2,3,4
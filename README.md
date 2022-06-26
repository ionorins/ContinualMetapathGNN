# Dynamic Learning for Recommender Systems

This project is a fork of the [blindsubmission1/PEAGNN](https://github.com/blindsubmission1/PEAGNN) repository. The main bulk of modifications are present in the following files:
- `experiments/peagat_solver_bpr.py`
- `graph_recsys_benchmark/models/peagat.py`
- `graph_recsys_benchmark/models/base.py`
- `graph_recsys_benchmark/solvers.py`
- `graph_recsys_benchmark/datasets/movielens.py`

An example command for running the project from the `experimets` directory:
```
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=latest-small --num_core=10 --num_feat_core=10 --sampling_strategy=unseen --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=3 --runs=1 --epochs=30 --batch_size=1024 --save_every_epoch=40 --metapath_test=false --num_timeframes=25 --equal_timespan_timeframes=false --ewc_lambda=0 --theta=1.5 --continual_aspect=continual --future_testing=true --train_between_emb_diff=false 
``` 

The history of the HR and nDCG over timeframes will be saved in the HR directory.

Below the description of the installation and of file structure of the project available in the original README file is reproduced.

## Requirements and Installation
* Python 3.6
* [PyTorch](http://pytorch.org/) 1.5.1
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 1.5.0
* Install all dependencies run
```
pip3 install -r requirements.txt
```


## Directory structure

The basic skeleton of our source code will look like this :
```bash

├── datasets
│   └── Yelp
│      └──  yelp_dataset.tar
├── experiments
│   ├── checkpoint
│   │   ├── data
│   │   │   ├── Movielenslatest-small
│   │   │   ├── Movielenslatest-25m
│   │   │   └── Yelp
│   │   ├── loggers
│   │   │   ├── Movielenslatest-small
│   │   │   ├── Movielenslatest-25m
│   │   │   └── Yelp
│   │   └── weights
│   │       ├── Movielenslatest-small
│   │       ├── Movielenslatest-25m
│   │       └── Yelp
│   ├── scripts
│   │   ├── **/*.ps1
│   └── **/*.py
├── graph_recsys_benchmark
│   ├── datasets
│   │   ├── **/*.py
│   ├── models
│   │   ├── **/*.py
│   ├── nn
│   │   ├── **/*.py
│   ├── parser
│   │   ├── **/*.py
│   ├── utils
│   │   ├── **/*.py
│   └── **/*.py
├── images
│   └── **/*.png
├── license.txt
├── README.md
├── requirements.txt
└── setup.py
```

## Description of the Code
The code is based on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) documentation. 

- [`experiments`](experiments): contain experiment files for PEAGNN and baseline models
- [`checkpoint`](experiments/checkpoint): contain processed data, logs and model weights
- [`scripts`](experiments/scripts): scripts to reproduce the results for each dataset
- [`datasets`](graph_recsys_benchmark/datasets): creates Heterogenous Information network for the datasets
- [`models`](graph_recsys_benchmark/models): creates PEAGNN and baseline models 
- [`nn`](graph_recsys_benchmark/nn): contains convolutional networks for GNN based models
- [`parser`](graph_recsys_benchmark/parser): functions to parse the raw dataset files
- [`utils`](graph_recsys_benchmark/utils): functions to save, load models and compute evaluation metrics

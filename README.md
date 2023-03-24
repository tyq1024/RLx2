# RLx2: Training a Sparse Deep Reinforcement Learning Model from Scratch

Code for the paper ["RLx2: Training a Sparse Deep Reinforcement Learning Model from Scratch"](https://openreview.net/forum?id=DJEEqoAq7to).

The DST Scheduler is inplemented based on an open-source PyTorch version of [RigL](https://github.com/nollied/rigl-torch) codebase. We implement the RL algorithms based on the official codebase of [TD3](https://github.com/sfujim/TD3) and an open-source PyTorch implementation of [SAC](https://github.com/denisyarats/pytorch_sac).

We use [MuJoCo](https://mujoco.org/) 2.0.0 from [OpenAI gym](https://github.com/openai/gym) for our experiments.

## Overview
```
├── DST                         //Modules for topology evolution
│   ├── __init__.py
│   ├── DST_Scheduler.py        //Scheduler for topology evolution
│   └── utils.py                //Other modules used      
├── RLx2_SAC
│   ├── train.py                    //Train with SAC
│   ├── SAC.py                      //SAC with n-step
│   └── modules_SAC.py              //Neural Networks
├── RLx2_TD3
│   ├── train.py                    //Train with TD3
│   ├── TD3.py                      //TD3 with n-step
│   └── modules_TD3.py              //Neural Networks
├── conda_env.yml 
└── README.md
```

## Usage

create conda environment:

```
conda env create -f conda_env.yml
conda activate RLx2
```

To run RLx2 in each single environment with TD3:

```
cd RLx2_TD3
python train.py --env <environment_name> --actor_sparsity <actor_sparsity> --critic_sparsity <critic_sparsity> --nstep 3 --delay_nstep 300000 --use_dynamic_buffer
```
To run RLx2 in each single environment with SAC:

```
cd RLx2_SAC
python train.py --env <environment_name> --actor_sparsity <actor_sparsity> --critic_sparsity <critic_sparsity> --nstep 3 --delay_nstep 300000 --use_dynamic_buffer
```

## Cite

```
@inproceedings{
tan2023rlx,
title={{RL}x2: Training a Sparse Deep Reinforcement Learning Model from Scratch},
author={Yiqin Tan and Pihe Hu and Ling Pan and Jiatai Huang and Longbo Huang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=DJEEqoAq7to}
}
```

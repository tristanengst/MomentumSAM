# NoNormSAM
*Authors*: Mehran Aghabozorgi, Tristan Engst, Oliver Fujiki, Yanshu Zhang
*Link to paper*: [...]()
Many thanks to David Samuel for creating the [repo](https://github.com/davda54/sam) we base ours on.

# Setup
```
conda create -n py310NNSAM python=3.10
conda activate py310NNSAM
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge tqdm wandb
```

# Training
```
python Train.py --opt OPTIMIZER
```

# ADL-Project

### Setup

1. Create a conda environment
```zsh
conda create -n prj-adl
conda activate prj-adl
```

2. Install PyTorch follwing the [official instructions](https://pytorch.org/get-started/locally/).
eg. for CUDA $\geq$ 12.4, using conda:
```zsh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. Install the requirements from the `requirements.txt` file
```zsh
pip install -r requirements.txt
```

4. Install the project's packages
```zsh
pip install -e src/litutils
```

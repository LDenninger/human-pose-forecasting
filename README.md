# Human Pose Forecasting
This projects aims at forecasting human poses with inspiraction from: https://arxiv.org/pdf/2004.08692.pdf
This is the final project of the lab course "Vision Systyems" of the University of Bonn.

## Installation
The complete project was run in a conda environment. Here we share the setup:

0. Initialize new conda environment: `conda create -n hpf python=3.11 `
1. Install PyTorch according to: https://pytorch.org/get-started/locally/ <br/>
    Additionally install PyTorch3d: `conda install -c pytorch3d pytorch3d`

2. Install Pip in conda environment: `conda install -c anaconda pip`
3. Install all additional packages with pip:
``` pip install optuna jupyter ipdb scipy tqdm matplotlib tensorboard wandb```

Before running any commands it is advised to source our small environment to enable shortcuts for experiment management: `source env.sh`

## Testing
We implemented different testing class to test and demonstrate the different components of our project: <br/>
Test and inspect the data: `python run.py --test_data` <br/>
Test and inspect the skeleton model against the baseline from MotionMixer: `python run.py --test_sk32` <br/>

## Experiment Management
Initially please run: `source env.sh` <br/>
Current experiment and run name are stored as environment variables, such that we can easily manipulate and work with a run. 
If the environment variables are set, one can ommit specifying the experiment or run. <br/>
Show current experiment setup: `setup` <br/>
Set experiment name: `setexp [exp. name]` <br/>
Set run name: `setrun [run name]` <br/>
Initialize new experiment: `iexp -exp [exp. name]` <br/>
Initialize new run: `irun -exp [exp. name] -run [run name]` <br/>
Clear tensorboard logs from run: `cllog -exp [exp. name] -run [run name]` <br/>
Train a model: `train -exp [exp. name] -run [run name]` <br/>

# Ideas
1. Training with different losses
    * Test out losses define of the joint positions instead of the rotation matrices
    * Use some kind of angular loss
    * Adversial loss
2. Predicting directly rotation matrix does not lead to numerical instabilities?? Possible SVD afterwards?
3. Test efficiency for embedded systems
    * Optimize inference with TensorRT and run on embedded system
    * Jetson AGX Nano?
    * How far can we watch in the future in real-time ?

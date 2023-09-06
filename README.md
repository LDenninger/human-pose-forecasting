# Human Pose Forecasting
This project aims at forecasting human poses with inspiration from: https://arxiv.org/pdf/2004.08692.pdf
This is the final project of the lab course "Vision Systems" of the University of Bonn.

## Project Structure

    .
    ├── ...
    ├── configurations                  # Configurations files
    ├── data                            # Dataset
    ├── experiments                     # Experiment directory
    ├── src                             # Python source files
    │   ├── data_utils                  # Functions and modules for data processing
    │   ├── evaluation                  # Functions and modules for the evaluation
    │   ├── models                      # Torch models used within the project
    │   ├── tests                       # Tests to demonstrate the correct behaviour
    │   ├── utils                       # Utility functions
    │   ├── visualization               # Visualization functions
    │   └── training_baseline.py        # Trainer for the baseline model
    ├── env.sh                          # Source file for the experiment environment
    ├── keep_exp.sh                     # script to add .gitignore files into empty folders in the experiment directory
    ├── run.py                          # General run file
    └── ...

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

# ToDo

    1) PyTorch dataloader for the AIS dataset
    2) Long-term prediction evaluation in the EvaluationEngine
        * here we might need a slightly different dataloader
        * implement the distribution metrics
    3) Implement the s22 skeleton
        * H36M skeleton with all joints removed that do not move
        * we have to look which joints exactly dont move
    4) Conversion between s26 and the other skeletons
    5) Add ability for absolute positions that are not centered at the hip
        * for rotation: add hip position as the first joint
        * for positions: use absolute positions
        * Here we might need to adapt the computation of losses and metrics since the position has another dimension
    6) Add possibility to evaluate the metrics on a per-joint basis
        * Using noise on joints evaluate how good our model predicts previously not seen joints
    7) Learned positional encodings
    8) Separate positional encoding for temporal and spatial domain
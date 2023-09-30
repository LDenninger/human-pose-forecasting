# Human Pose Forecasting

This project was developed as the final lab project for the lab "Vision Systems" at the University of Bonn: https://www.ais.uni-bonn.de/SS23/4308_Lab_Vision_Systems.html

Within this project we implemented the Spatio-Temporal Transformer according to: https://arxiv.org/pdf/2004.08692.pdf

The complete model was build from scratch using the PyTorch framework. 

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
        │   └── Session.py                  # Session encapsuling all training and evaluation
        ├── env.sh                          # Source file for the experiment environment
        ├── keep_exp.sh                     # script to add .gitignore files into empty folders in the experiment directory
        ├── run.py                          # General run file
        ├── train.py                        # Run a training
        ├── test.py                         # Test specific parts of the project
        ├── evaluate.py                     # Evaluate a trained model from the experiment directory
        ├── visualize.py                    # Produce visualizations using a trained model
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


## Experiment Structure

```
            .
        ├── ...
        ├── experiments                     # Experiment directory
        │   ├── example_experiment          # Example experiment
        │   │   ├── example_run             # Example run
        │   │   │   ├── checkpoints         # Directory holding the checkpoints made during training
        │   │   │   ├── logs                # Contains all logs written during training and evaluation
        │   │   │   ├── visualizations      # Contains all visualization made for evaluation
        │   │   │   ├── config.json         # Contains all configuration parameters for the model and training
        │   │   └── ...
        │   └── ...    
        └── ...       
```
## Model Implementations

We implemented all models from scratch according to: https://arxiv.org/abs/2004.08692. <br/>
For some extended details we also considered the original implementation in Tensorflow V1 from: https://github.com/eth-ait/motion-transformer. 

The implemented PyTorch modules are structured in three stages.

**Attention**: At the lowest level we implemented a vanilla, spatial and temporal attention mechanism. The code can be found at: [src/models/attention.py](src/models/attention.py) 

**Transformer**: Next we implemented the used transformer blocks that implement the attention mechanism. This includes a vanilla, spatio-temporal and two sequential transformer blocks that were used for our experiments. The code can be found at: [src/models/transformer.py](src/models/transformer.py)

**Pose Predictor**: Finally, we implemented a pose predictor that can use different transformer blocks and architectures to auto-regressively predict poses. The code can be found at: [src/models/PosePredictor.py](src/models/PosePredictor.py)

**Additional Modules**: The positional encoding can be found in [src/models/positional_encoding.py](src/models/positional_encoding.py). Additional processing functions and modules are located at: [src/models/utils.py](src/models/utils.py)

## Contact
For any question, feel free to contact me: Luis0512@web.de

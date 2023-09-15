import torch
import argparse
import os
from typing import List

from src import Session
from src.utils import print_ 

from src.data_utils.data_loader import *
from src.data_utils.data_loading import *

# Get basepath
basepath = os.getcwd()

# Load base config file
config = json.load(open(basepath + "/configurations/baseline_config.json", "r"))

# Get dataset
dataset = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = "s16", is_train = True, debug = False)
mean, var = dataset.get_mean_variance()

# Save the two tensors
torch.save(mean, basepath + "/configurations/mean.pt")
torch.save(var, basepath + "/configurations/var.pt")

# Load tensors and print
mean = torch.load(basepath + "/configurations/mean.pt")
var = torch.load(basepath + "/configurations/var.pt")
print(mean)
print(var)
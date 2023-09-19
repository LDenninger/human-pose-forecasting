import os
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.data_loader import *
from src.evaluation.metrics import *

if __name__ == "__main__":
    # Get basepath
    basepath = os.getcwd()

    # Load base config file
    config = json.load(open(basepath + "/configurations/baseline_config.json", "r"))

    # Get dataset
    dataset = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = "s16", is_train = False, debug = False)

    # Create DataLoader
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=False,
        )
    
    # Iterate over DataLoader and calculate the average entropy of the dataset
    entropy = 0
    print("Calculating entropy of dataset...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Calculate power spectrum of batch
        ps = power_spectrum(batch) # (n_joints, seq_len, feature_size)
        # Calculate entropy of power spectrum
        ps_ent = ps_entropy(ps) # (feature_size,)
        # Calculate average entropy of power spectrum
        avg = torch.mean(ps_ent)
        entropy += avg
    entropy /= len(dataloader)


    print(entropy)
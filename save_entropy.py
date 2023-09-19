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
    config = json.load(open(basepath + "/experiments/entropy_study/60seconds/config.json", "r"))

    # Get dataset
    train_set = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = "s16", is_train = True, debug = False)
    test_set = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = "s16", is_train = False, debug = False)

    # Create DataLoader
    train = DataLoader(
            dataset=train_set,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=True,
        )
    test = DataLoader(
            dataset=test_set,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=True,
        )

    # Iterate over DataLoader and calculate the average entropy of the test dataset
    entropy = 0
    print("Calculating entropy of training dataset...")
    for i, batch in tqdm(enumerate(test), total=len(test)):
        # Calculate power spectrum of batch
        ps = power_spectrum(batch) # (n_joints, seq_len, feature_size)
        # Calculate entropy of power spectrum
        ps_ent = ps_entropy(ps) # (feature_size,)
        # Calculate average entropy of power spectrum
        avg = torch.mean(ps_ent)
        entropy += avg
    entropy /= len(test)
    # Save entropy as torch tensor
    torch.save(entropy, basepath + "/configurations/entropy.pt")

    # Calculate symmetric kl-divergence between training and test dataset
    kld = 0
    print("Calculating symmetric KL-divergence between training and test dataset...")
    for i, (train_batch, test_batch) in tqdm(enumerate(zip(train, test)), total=min(len(train), len(test))):
        # Calculate power spectrum of batch
        train_ps = power_spectrum(train_batch)
        test_ps = power_spectrum(test_batch)
        # Calculate kl-divergence in both directions
        kl1 = ps_kld(train_ps, test_ps)
        kl2 = ps_kld(test_ps, train_ps)
        # Calculate average kl-divergence
        kl1 = torch.mean(kl1)
        kl2 = torch.mean(kl2)
        # Calculate symmetric kl-divergence
        kld += kl1 + kl2
    kld /= 2 * (i + 1)
    # Save kld as torch tensor
    torch.save(kld, basepath + "/configurations/kld.pt")
    print(entropy)
    print(kld)
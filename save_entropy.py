import os
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.data_loader import *
from src.evaluation.metrics import *

if __name__ == "__main__":
    
    SKELETON_MODEL = "s16"
    N_TOTAL_SAMPLES = 20000
    ABSOLUTE_POSITION = False
    NORMALIZE_ORIENTATION = True
    DATASET = "h36m"

    # Get basepath
    basepath = os.getcwd()

    # Load base config file
    config = json.load(open(basepath + "/experiments/entropy_study/60seconds/config.json", "r"))

    # Set orienation normalization
    config["dataset"]["normalize_orientation"] = NORMALIZE_ORIENTATION
    config["dataset"]["name"] = DATASET

    # Get dataset
    train_set = getDataset(config = config["dataset"], absolute_position=ABSOLUTE_POSITION, joint_representation = "pos", skeleton_model = SKELETON_MODEL, is_train = True, debug = False)
    test_set = getDataset(config = config["dataset"], absolute_position=ABSOLUTE_POSITION, joint_representation = "pos", skeleton_model = SKELETON_MODEL, is_train = False, debug = False)

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
    representation = config['joint_representation']['type']
    def collect_samples(dataset):
        all_samples = []
        i = 0
        while i < N_TOTAL_SAMPLES:
            for batch in dataset:
                if representation != 'pos':
                    batch, _ = h36m_forward_kinematics(batch, representation)
                all_samples.append(batch)
                i += batch.shape[0]
                if i >= N_TOTAL_SAMPLES:
                    break
        return torch.cat(all_samples, dim=0)[:N_TOTAL_SAMPLES]
    
    
    train_samples = collect_samples(train)
    test_samples = collect_samples(test)

    train_ps = power_spectrum(train_samples)
    test_ps = power_spectrum(test_samples)
    
    train_ent = ps_entropy(train_ps)
    test_ent = ps_entropy(test_ps)

    print(f"Train entropy: {(torch.mean(train_ent))}")
    print(f"Test entropy: {(torch.mean(test_ent))}")

    train_kld = ps_kld(train_ps, test_ps)
    test_kld = ps_kld(test_ps, train_ps)

    print(f"Train KLD: {(torch.mean(train_kld))}")
    print(f"Test KLD: {(torch.mean(test_kld))}")

    entropy = torch.mean(test_ent)
    kld = (torch.mean(train_kld) + torch.mean(test_kld)) / 2

    if not os.path.exists(basepath + "/configurations/distribution_values"):
        os.makedirs(basepath + "/configurations/distribution_values")

    torch.save(kld, basepath + f"/configurations/distribution_values/kld_{SKELETON_MODEL}_{DATASET}_norm_{NORMALIZE_ORIENTATION}_abs_{ABSOLUTE_POSITION}.pt")
    torch.save(entropy, basepath + f"/configurations/distribution_values/entropy_{SKELETON_MODEL}_{DATASET}_norm_{NORMALIZE_ORIENTATION}_abs_{ABSOLUTE_POSITION}.pt")
    torch.save(test_ps, basepath + f"/configurations/distribution_values/test_ps_{SKELETON_MODEL}_{DATASET}_norm_{NORMALIZE_ORIENTATION}_abs_{ABSOLUTE_POSITION}.pt")
    print(entropy)
    print(kld)

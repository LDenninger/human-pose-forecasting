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
    
    # Get basepath
    basepath = os.getcwd()

    # Load base config file
    config = json.load(open(basepath + "/experiments/entropy_study/60seconds/config.json", "r"))

    # Get dataset
    train_set = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = SKELETON_MODEL, is_train = True, debug = False)
    test_set = getDataset(config = config["dataset"], joint_representation = "pos", skeleton_model = SKELETON_MODEL, is_train = False, debug = False)

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

    torch.save(kld, basepath + f"/configurations/distribution_values/kld_{SKELETON_MODEL}.pt")
    torch.save(entropy, basepath + f"/configurations/distribution_values/entropy_{SKELETON_MODEL}.pt")
    torch.save(test_ps, basepath + f"/configurations/distribution_values/test_ps_{SKELETON_MODEL}.pt")
    print(entropy)
    print(kld)
    
    
    # Iterate over DataLoader and calculate the average entropy of the test dataset
    # entropy = 0
    # print("Calculating entropy of training dataset...")
    # for i, batch in tqdm(enumerate(test), total=len(test)):
    #     # Use joint position as representation to make models operating on different representations comparable
    #     if representation != 'pos':
    #         batch, _ = h36m_forward_kinematics(batch, representation)
    #     # Calculate power spectrum of batch
    #     ps = power_spectrum(batch) # (n_joints, seq_len, feature_size)
    #     # Calculate entropy of power spectrum
    #     ps_ent = ps_entropy(ps) # (feature_size,)
    #     # Calculate average entropy of power spectrum
    #     avg = torch.mean(ps_ent)
    #     entropy += avg
    # entropy /= len(test)
    # # Save entropy as torch tensor

    # # Calculate symmetric kl-divergence between training and test dataset
    # kld = 0
    # print("Calculating symmetric KL-divergence between training and test dataset...")
    # for i, (train_batch, test_batch) in tqdm(enumerate(zip(train, test)), total=min(len(train), len(test))):
    #     # Use joint position as representation to make models operating on different representations comparable
    #     if representation != 'pos':
    #         train_batch, _ = h36m_forward_kinematics(train_batch, representation)
    #         test_batch, _ = h36m_forward_kinematics(test_batch, representation)
    #     # Calculate power spectrum of batch
    #     train_ps = power_spectrum(train_batch)
    #     test_ps = power_spectrum(test_batch)
    #     # Calculate kl-divergence in both directions
    #     kl1 = ps_kld(train_ps, test_ps)
    #     kl2 = ps_kld(test_ps, train_ps)
    #     # Calculate average kl-divergence
    #     kl1 = torch.mean(kl1)
    #     kl2 = torch.mean(kl2)
    #     # Calculate symmetric kl-divergence
    #     kld += kl1 + kl2
    # kld /= 2 * (i + 1)
    # Save kld as torch tensor

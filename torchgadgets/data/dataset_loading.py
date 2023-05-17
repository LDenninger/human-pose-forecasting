import torch
from torch.utils.data import DataLoader

import torchvision as tv
import torchvision.datasets as datasets

def get_train_loaders(config):
    data = load_dataset(dataset_name=config['dataset']['name'])
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    return train_loader, test_loader

def load_dataset(dataset_name, train_dataset=True, test_dataset=True, val_dataset=False):
    """
        Load the train, test, and/or the validation set for a given dataset from torchvision.
    
    """
    dataset = {}

    if train_dataset:
        dataset['train_dataset'] = _load(dataset_name, mode=0)
    if test_dataset:
        dataset['test_dataset'] = _load(dataset_name, mode=1)
    if val_dataset:
        dataset['val_dataset'] = _load(dataset_name, mode=2)

    return dataset

def extract_dataset(dataset: torch.utils.data.Dataset, toTensor: bool = True):
    """
        Extracts the data from a dataset.

        Arguments:
            dataset (torch.utils.data.Dataset): PyTorch dataset to extract the data from to use it in our custom dataset or directly work on the data.

    """

    data = []
    labels = []
    
    for i, (d, l) in enumerate(dataset):
        if toTensor and not torch.is_tensor(d):
            d = tv.transforms.PILToTensor()(d)
        data.append(d)
        labels.append(l)


    return data, labels


    

def _load(dataset_name, mode=0):
    """
        Loads a split from an arbitray dataset from torchvision.

        Arguments:
            dataset_name (str): Name of the dataset to load.
            mode (int): 0 for train, 1 for test, 2 for val.
    
    """
    train_dataset = None


    if dataset_name == 'cifar10':
        if mode == 2:
            print(f'No validation dataset available for {dataset_name}')
            return None
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True if mode==0 else False, download=True)
    elif dataset_name == 'cifar100':
        if mode == 2:
            print(f'No validation dataset available for {dataset_name}')
            return None
        train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True if mode==0 else False, download=True)
    elif dataset_name == 'mnist':
        if mode == 2:
            print(f'No validation dataset available for {dataset_name}')
            return None
        train_dataset = datasets.MNIST(root='./data/mnist', train=True if mode==0 else False, download=True)
    elif dataset_name == 'fashionmnist':
        if mode == 2:
            print(f'No validation dataset available for {dataset_name}')
            return None
        train_dataset = datasets.FashionMNIST(root='./data/fashionmnist', train=True, download=True)
    elif dataset_name =='svhn':
        if mode == 2:
            s = 'extra'
        if mode == 0:
            s = 'train'
        else:
            s = 'test'
        train_dataset = datasets.SVHN(root='./data/svhn', split=s, download=True)
    elif dataset_name == 'oxfordpet':
        if mode == 2:
            print(f'No validation dataset available for {dataset_name}')
            return None
        train_dataset = datasets.OxfordIIITPet(root='./data/oxfordpet', split='trainval' if mode==0 else 'test', download=True)
    if train_dataset is None:
        print(f'No dataset available for {dataset_name}')
    
    return train_dataset

import torch 
import torchvision as tv

import numpy as np

from functools import reduce

from .dataset_loading import extract_dataset

import copy


class ImageDataset(torch.utils.data.Dataset):
    """
        A simple wrapper class for PyTorch datasets to add some further functionalities.
    
    """
    def __init__(self, dataset: torch.utils.data.Dataset, transforms: list = None, train_set: bool = True):
        self.dataset = dataset
        self.data_augmentor = None
        self.train_set = train_set
        self.filter = False

        if transforms is not None:
            self.data_augmentor = ImageDataAugmentor(transforms)

    def __getitem__(self, index):
        image, label = self.dataset[index] if not self.filter else (self.dataset[self.filter_ind[index]][0], self.filter_labels[self.filter_inds[index]])
        if self.data_augmentor is not None:
            image, label = self.data_augmentor((image, label), self.train_set)
        return image, label
    
    def remap_labels(self, mapping: dict):
        if self.filter:
            self.reset_filter()
        self.filter = True
        for i, label in enumerate(self.filter_labels):
            self.filter_labels[i] = mapping[str(label)]
    def filter_dataset(self, inds: list):
        self.filter = True
        self.filter_inds = inds
    
    def reset_filter(self):
        self.filter = False
        self.filter_labels = copy.deepcopy(self.labels)
        self.filter_inds = torch.arange(0, len(self.labels))
    
    def __len__(self):
        return len(self.dataset) if not self.filter else len(self.filter_inds)

"""
    Data Augmentation Module:
        Data augmentation steps are provided as a list of dictionaries. Each dictionary is a description of a data augmentation step.
        The possible keys are:
            {'type': 'flatten', 'train': True, 'eval': True},
            {'type': 'rgb2gray', 'train': True, 'eval': True},
            {'type': 'normalize', 'train': True, 'eval': True},
            {'type': 'permute', 'train': True, 'eval': True},
            {'type': 'rotate', 'degrees': 90, 'train': True, 'eval': True},
            {'type': 'resize', 'size': (height, width), 'train': True, 'eval': True},
            {'type': 'random_rotation', 'degrees': 90, 'train': True, 'eval': True},
            {'type': 'random_crop', 'size': (height, width), 'train': True, 'eval': True},
            {'type': 'random_horizontal_flip', 'p': 0.5, 'train': True, 'eval': True},
            {'type': 'random_erase', 'probability': 0.5, 'scale': (0.01,0.2), 'train': True, 'eval': True},

"""

class ImageDataAugmentor:
    """
        Module to apply pre-defined data augmentations on batches of images as well as single images.
    
    """

    def __init__(self, config: list):
        self.config = config
        self.alternative_input = None
        self._init_pipeline()

    def __call__(self, image, train=True):
        return self.train_pipeline(image) if train else self.eval_pipeline(image)
    
    def set_alternative_input(self, input: tuple):
        self.alternative_input = input
    
    def processing_pipeline(self, *funcs):
        """
            Returns a function that applies a sequence of data augmentation steps in a pipeline.
        """
        return lambda x: reduce(lambda acc, f: f(acc), funcs, x)
    
    ## Processing Functions ##    
    def _flatten_img(self, input: tuple):
        # Flatten only the image size dimensions
        if self.flatten_only_img_size:
            return torch.flatten(input, start_dim=-2)
    
        # Flatten all dimensions except of the batch dimension
        else:
            return torch.flatten(input, start_dim=1)

    def _rgb2grayscale(self, input: tuple):
        return (self.gray_converter(input[0]), input[1])
    
    def _normalize(self, input: tuple):
        if input[0].shape[1] == 1 or input[0].shape[-1] == 1:
            return (self.normalizer_gray(input[0]), input[1])
        return (self.normalizer_rgb(input[0]), input[1])
    
    def _permute(self, input: tuple):
        if len(input[0].shape) == 4 and len(self.permute_dim.shape)==3:
            p_dim = (0, self.permute_dim[0], self.permute_dim[1], self.permute_dim[2])
            return (input[0].permute(p_dim), input[1])
        return (input[0].permute(self.permute_dim), input[1])
    
    def _resize(self, input: tuple):
        return (self.resizer(input[0]), input[1])
    
    def _random_rotation(self, input: tuple):
        return (self.random_rotater(input[0]), input[1])
    
    def _random_horizontal_flip(self, input: tuple):
        return (self.random_horizontal_flipper(input[0]), input[1])
    
    def _random_vertical_flip(self, input: tuple):
        return (self.random_vertical_flipper(input[0]), input[1])
    
    def _gaussian_blur(self, input: tuple):
        return (self.gaussian_filter(input[0]), input[1])

    def _random_erase(self, input: tuple):
        return (self.random_eraser(input[0]), input[1])
    
    def _random_color_jitter(self, input: tuple):
        return (self.color_jitter(input[0]), input[1])
    
    def _random_crop(self, input: tuple):
        return (self.random_cropper(input[0]), input[1])

    def _dynamic_random_crop(self, input: tuple):
        """
            Performs a random crop on a single image with a dynamic crop size. 
            If an image dimension is smaller than the crop size, it is cropped with a crop size spanning the maximum of the image height and width.
        """
        assert len(input[0].shape) == 3, 'Please provide a single image in a 3D tensor. Provided image shape: {}'.format(input[0].shape)

        if input[0].shape[0] in [1,2,3,4]:
            C, H, W = input[0].shape
        else:
            H, W, C = input[0].shape
        crop_size = (self.crop_size[0], self.crop_size[1])
        if H<crop_size[0]:
            crop_size = (H, H)
        if W<crop_size[1]:
            if W<crop_size[0]:
                crop_size = (W, W)
        return (tv.transforms.RandomCrop(crop_size)(input[0]), input[1])
    
    def _random_resized_crop(self, input: tuple):
        return (self.random_resized_cropper(input[0]), input[1])
    
    def _center_crop(self, input: tuple):
        return (self.center_cropper(input[0]), input[1])
    
    def _dynamic_center_crop(self, input: tuple):
        """
            Performs a (quadratic) center crop on a single image with a dynamic crop size.
            The crop size is the minimum of the image height and width.
        """
        assert len(input[0].shape) == 3, 'Please provide a single image in a 3D tensor. Provided image shape: {}'.format(input[0].shape)
        if input[0].shape[0] in [1,2,3,4]:
            C, H, W = input[0].shape
        else:
            H, W, C = input[0].shape
        crop_size = min([H, W])
        crop_size = (crop_size, crop_size)
        return (tv.transforms.CenterCrop(crop_size)(input[0]), input[1])

    def _adjust_brightness(self, input: tuple):
        return (tv.transforms.functional.adjust_brightness(input[0], self.brightness_factor), input[1])
    
    def _adjust_contrast(self, input: tuple):
        return (tv.transforms.functional.adjust_contrast(input[0], self.contrast_factor), input[1])
    
    def _adjust_saturation(self, input: tuple):
        return (tv.transforms.functional.adjust_saturation(input[0], self.saturation_factor), input[1])
    
    def _adjust_hue(self, input: tuple):
        return (tv.transforms.functional.adjust_hue(input[0], self.hue_factor), input[1])
    
    def _adjust_gamma(self, input: tuple):
        return (tv.transforms.functional.adjust_gamma(input[0], self.gamma_factor), input[1])
    
    def _adjust_sharpness(self, input: tuple):
        return (tv.transforms.functional.adjust_sharpness(input[0], self.sharpness_factor), input[1])
    
    def _squeeze(self, input: tuple):
        if self.squeeze_dim is not None:
            return (input[0].squeeze(dim=self.squeeze_dim), input[1])
        return (input[0].squeeze(), input[1])
    
    def _convert_255_to_1(self, input: tuple):
        return (input[0] / 255., input[1])
    
    def _toTensor(self, input: tuple):
        return (self.tensorTransformer(input[0]), input[1])
    
    def _blank(self, input: tuple):
        return input
    
    def _mixup(self, input: tuple):
        """
            Implementation of the data augmentation technique as described in: "mixup: beyond empirical risk minimization" <https://arxiv.org/pdf/1710.09412.pdf>
        """

        device = input[0].device
        rand_index = torch.randperm(input[0].shape[0]).to(device)
        target_pairs = input[1][rand_index]
        batch_pairs = input[0][rand_index]

        target_ohc = torch.nn.functional.one_hot(input[1], num_classes=self.mixup_classes)
        target_pairs_ohc = torch.nn.functional.one_hot(target_pairs, num_classes=self.mixup_classes)

        lam = self.mixup_beta.sample()

        return (lam * input[0] + (1 - lam) * batch_pairs, lam * target_ohc + (1 - lam) * target_pairs_ohc)
    
    def _cutmix(self, input: tuple):
        """
            Implementation of the data augmentation technique as described in: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" <https://arxiv.org/pdf/1905.04899.pdf>
            This code is an adaptation from: https://github.com/clovaai/CutMix-PyTorch
        """

        def _rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2
        device = input[0].device
        # generate mixed sample
        lam = self.cutmix_beta.sample()
        rand_index = torch.randperm(input[0].shape[0]).to(device)
        target_pairs = input[1][rand_index]
        batch_pairs = input[0][rand_index]
        bbx1, bby1, bbx2, bby2 = _rand_bbox(batch_pairs.shape, lam)
        batch_pairs[:, :, bbx1:bbx2, bby1:bby2] = batch_pairs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input[0].shape[-1] * input[0].shape[-2]))
        target = torch.nn.functional.one_hot(input[1], num_classes=self.cutmix_classes) * lam + (1 - lam) * torch.nn.functional.one_hot(target_pairs, num_classes=self.cutmix_classes)
        return (batch_pairs, target)
  
    def _init_pipeline(self):
        train_augmentation = []
        eval_augmentation = []
        for process_step in self.config:
            if process_step['type'] == 'flatten_img':
                if process_step['train']:
                    train_augmentation.append(self._flatten_img)
                if process_step['eval']:
                    eval_augmentation.append(self._flatten_img)
            elif process_step['type'] == 'rgb2gray':
                self.gray_converter = tv.transforms.Grayscale()
                if process_step['train']:
                    train_augmentation.append(self._rgb2grayscale)
                if process_step['eval']:
                    eval_augmentation.append(self._rgb2grayscale)
            elif process_step['type'] == 'normalize':
                self.normalizer_gray = tv.transforms.Normalize(mean=[0.485], std=[0.229])
                self.normalizer_rgb = tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                if process_step['train']:
                    train_augmentation.append(self._normalize)
                if process_step['eval']:
                    eval_augmentation.append(self._normalize)
            elif process_step['type'] == 'permute':
                self.permute_dim = process_step['dim']
                if process_step['train']:
                    train_augmentation.append(self._permute)
                if process_step['eval']:
                    eval_augmentation.append(self._permute)
            elif process_step['type'] =='resize':
                self.resizer = tv.transforms.Resize(process_step['size'])
                if process_step['train']:
                    train_augmentation.append(self._resize)
                if process_step['eval']:
                    eval_augmentation.append(self._resize)
            elif process_step['type'] == 'random_rotation':
                self.random_rotater = tv.transforms.RandomRotation(process_step['degrees'])
                if process_step['train']:
                    train_augmentation.append(self._random_rotation)
                if process_step['eval']:
                    eval_augmentation.append(self._random_rotation)
            elif process_step['type'] == 'random_crop':
                self.random_cropper = tv.transforms.RandomCrop(process_step['size'])
                if process_step['train']:
                    train_augmentation.append(self._random_crop)
                if process_step['eval']:
                    eval_augmentation.append(self._random_crop)
            elif process_step['type'] == 'gaussian_blur':
                self.gaussian_filter = tv.transforms.GaussianBlur(kernel_size=process_step['kernel_size'], sigma = process_step['sigma'])
                if process_step['train']:
                    train_augmentation.append(self._gaussian_blur)
                if process_step['eval']:
                    eval_augmentation.append(self._gaussian_blur)
            elif process_step['type'] == 'random_erase':
                self.random_eraser = tv.transforms.RandomErasing(p=process_step['p'], scale=process_step['scale'])
                if process_step['train']:
                    train_augmentation.append(self._random_erase)
                if process_step['eval']:
                    eval_augmentation.append(self._random_erase)
            elif process_step['type'] == 'adjust_brightness':
                self.brightness_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_brightness)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_brightness)
            elif process_step['type'] == 'adjust_contrast':
                self.contrast_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_contrast)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_contrast)
            elif process_step['type'] == 'adjust_saturation':
                self.saturation_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_saturation)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_saturation)
            elif process_step['type'] == 'adjust_hue':
                self.hue_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_hue)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_hue)
            elif process_step['type'] == 'adjust_gamma':
                self.gamma_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_gamma)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_gamma)
            elif process_step['type'] == 'color_jitter':
                self.color_jitter = tv.transforms.ColorJitter(brightness=process_step['brightness'], contrast=process_step['contrast'], saturation=process_step['saturation'], hue=process_step['hue'])
                if process_step['train']:
                    train_augmentation.append(self._random_color_jitter)
                if process_step['eval']:
                    eval_augmentation.append(self._random_color_jitter)
            elif process_step['type'] =='squeeze':
                self.squeeze_dim = process_step['dim']
                if process_step['train']:
                    train_augmentation.append(self._squeeze)
                if process_step['eval']:
                    eval_augmentation.append(self._squeeze)
            elif process_step['type'] == 'mixup':
                self.mixup_alpha = process_step['alpha']
                self.mixup_prob = process_step['prob']
                self.mixup_classes = process_step['num_classes']
                self.mixup_beta = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha)
                if process_step['train']:
                    train_augmentation.append(self._mixup)
                if process_step['eval']:
                    eval_augmentation.append(self._mixup)
            elif process_step['type'] == 'cutmix':
                self.cutmix_alpha = process_step['alpha']
                self.cutmix_prob = process_step['prob']
                self.cutmix_classes = process_step['num_classes']
                self.cutmix_beta = torch.distributions.beta.Beta(self.cutmix_alpha, self.cutmix_alpha)
                if process_step['train']:
                    train_augmentation.append(self._cutmix)
                if process_step['eval']:
                    eval_augmentation.append(self._cutmix)
            elif process_step['type'] =='dynamic_random_crop':
                self.crop_size = process_step['size']
                if process_step['train']:
                    train_augmentation.append(self._dynamic_random_crop)
                if process_step['eval']:
                    eval_augmentation.append(self._dynamic_random_crop)
            elif process_step['type'] == 'random_resized_crop':
                self.random_resized_cropper = tv.transforms.RandomResizedCrop(process_step['size'], process_step['scale'], process_step['ratio'])
                if process_step['train']:
                    train_augmentation.append(self._random_resized_crop)
                if process_step['eval']:
                    eval_augmentation.append(self._random_resized_crop)
            elif process_step['type'] == 'random_horizontal_flip':
                self.random_horizontal_flipper = tv.transforms.RandomHorizontalFlip(p=process_step['prob'])
                if process_step['train']:
                    train_augmentation.append(self._random_horizontal_flip)
                if process_step['eval']:
                    eval_augmentation.append(self._random_horizontal_flip)
            elif process_step['type'] == 'random_vertical_flip':
                self.random_vertical_flipper = tv.transforms.RandomVerticalFlip(p=process_step['prob'])
                if process_step['train']:
                    train_augmentation.append(self._random_vertical_flip)
                if process_step['eval']:
                    eval_augmentation.append(self._random_vertical_flip)
            elif process_step['type'] == 'center_crop':
                self.center_cropper = tv.transforms.CenterCrop(process_step['size'])
                if process_step['train']:
                    train_augmentation.append(self._center_crop)
                if process_step['eval']:
                    eval_augmentation.append(self._center_crop)
            elif process_step['type'] == 'dynamic_center_crop':
                if process_step['train']:
                    train_augmentation.append(self._dynamic_center_crop)
                if process_step['eval']:
                    eval_augmentation.append(self._dynamic_center_crop)
            elif process_step['type'] == 'convert_255_to_1':
                if process_step['train']:
                    train_augmentation.append(self._convert_255_to_1)
                if process_step['eval']:
                    eval_augmentation.append(self._convert_255_to_1)
            elif process_step['type'] == 'toTensor':
                self.tensorTransformer = tv.transforms.ToTensor()
                if process_step['train']:
                    train_augmentation.append(self._toTensor)
                if process_step['eval']:
                    eval_augmentation.append(self._toTensor)
        if len(train_augmentation) == 0:
            train_augmentation.append(self._blank)
        if len(eval_augmentation) == 0:
            eval_augmentation.append(self._blank)
        self.train_pipeline = self.processing_pipeline(*train_augmentation)
        self.eval_pipeline = self.processing_pipeline(*eval_augmentation)

def get_image_size(dataset):
    image_sizes = torch.zeros((len(dataset), 2))
    for i in range(len(dataset)):
        img = dataset[i][0]
        img = tv.transforms.PILToTensor()(img)
        _, height, width = img.shape
        image_sizes[i, 0] = height
        image_sizes[i, 1] = width

    return image_sizes

def compute_size_mean_std(dataset):
    image_sizes = get_image_size(dataset)
    mean_height = torch.mean(image_sizes[...,0], dim=0)
    mean_width = torch.mean(image_sizes[...,1], dim=0)
    std_height = torch.std(image_sizes[...,0], dim=0)
    std_width = torch.std(image_sizes[...,1], dim=0)
    return mean_height, mean_width, std_height, std_width
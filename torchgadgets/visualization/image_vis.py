import torch
import torchvision as tv

import matplotlib.pyplot as plt

import numpy as np

def visualize_dataset(dataset, data_augmentor=None, ind: list=None, label_names: list=None, layout= (4,4)):
    NUM_IMG = layout[0]*layout[1]

    fig, ax = plt.subplots(layout[0], layout[1])
    fig.set_size_inches(3 * layout[1], 3 * layout[0])
    if ind is None:
        ind = np.random.randint(low=0, high=len(dataset), size=NUM_IMG)
    if len(ax.shape) == 1:
        ax = ax[None]

    for i, n in enumerate(ind):
        img = dataset[n][0]
        if not torch.is_tensor(img):
            img = tv.transforms.PILToTensor()(img)
        if img.shape[0] in [1,2,3,4]:
            img = img.permute(1,2,0)
        label = dataset[n][1]

        y = i % layout[1]
        x = i // layout[1]
        if img.shape[-1]==1:
            ax[x][y].imshow(img, cmap='gray')

        else:
            ax[x][y].imshow(img)
        label = label_names[label] if label_names is not None else str(label)
        ax[x][y].set_title(f"Img #{n}  Label: {label}\nSize: H:{img.shape[0]}, W:{img.shape[1]}, C:{img.shape[2]}", fontsize=10)

    for x in range(ax.shape[0]):
        for y in range(ax.shape[1]):
            ax[x][y].axis("off")

def visualize_dataloader(dataset, data_augmentor=None, label_names: list=None, layout= (4,4)):
    NUM_IMG = layout[0]*layout[1]

    fig, ax = plt.subplots(layout[0], layout[1])
    fig.set_size_inches(3 * layout[1], 3 * layout[0])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (img, label) in enumerate(data_loader):
        if i == NUM_IMG:
            break
        if not torch.is_tensor(img):
            img = tv.transforms.PILToTensor()(img)
        if data_augmentor is not None:
            img, label = data_augmentor((img, label))
        img = img[0]
        label = label[0].tolist()
        img = img.permute(1,2,0)
        y = i % layout[1]
        x = i // layout[1]
        if img.shape[-1]==1:
            ax[x][y].imshow(img[0], cmap='gray')

        else:
            ax[x][y].imshow(img)
        
        ax[x][y].set_title(f"Label: {label}\nSize: H:{img.shape[0]}, W:{img.shape[1]}, C:{img.shape[2]}", fontsize=10)

    for x in range(ax.shape[0]):
        for y in range(ax.shape[1]):
            ax[x][y].axis("off")


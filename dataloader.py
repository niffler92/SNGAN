import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from nsml import HAS_DATASET, DATASET_PATH

from utils import *


def get_loader(
        dataset='CIFAR10',
        root='./data',
        batch_size=128,
        num_workers=4
        ):

    assert dataset in ['CIFAR10', 'CIFAR100']
    if DATASET_PATH:
        assert HAS_DATASET, "Can't find dataset in nsml. Push or search the dataset"
        root = os.path.join(DATASET_PATH, 'train')

    train_loader, val_loader = (torch.utils.data.DataLoader(
        globals()[dataset](root=root, train=is_training, download=True).preprocess(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
        ) for is_training in [True, False]
    )
    return train_loader, val_loader


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download=True):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize,])
        return self


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


class GenDataset(torch.utils.data.Dataset):
    """Dataset for Generator
    """
    def __init__(self, G, nsamples):
        self.G = G
        self.nsamples = nsamples
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        z = to_var(torch.randn(1, self.G.z_dim))
        return self.transform(np.squeeze(to_np(self.denorm(self.G(z)).permute(0, 2, 3, 1))))

    def __len__(self):
        return self.nsamples

    def denorm(self, x):
        # For fake data generated with tanh(x)
        x = (x + 1) / 2
        return x.clamp(0, 1)

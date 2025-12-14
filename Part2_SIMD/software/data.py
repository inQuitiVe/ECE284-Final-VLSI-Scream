import torch
import torchvision
import torchvision.transforms as transforms
from config import bargs


def get_data_loaders():
    """r
    CIFAR10 has 50,000 training data, and 10,000 validation data.
    Shape of data is (3, 32, 32)
    """
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            # transforms.RandomErasing(p=0.25),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_root = "./data"

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bargs.train_bsz,
        shuffle=True,
        num_workers=bargs.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=bargs.test_bsz,
        shuffle=False,
        num_workers=bargs.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return trainloader, testloader

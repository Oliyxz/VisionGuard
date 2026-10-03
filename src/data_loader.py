import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar100_dataloaders(data_dir='./data', batch_size=64, num_workers=2):
    """
    Load CIFAR-100 dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                             download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                            download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def get_tiny_imagenet_dataloaders(data_dir='./data/tiny-imagenet-200', batch_size=64, num_workers=2):
    """
    Load Tiny-ImageNet dataset using ImageFolder.
    Assumes the dataset is downloaded and the validation set is restructured.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Tiny-ImageNet directories not found at {data_dir}. Please run download_tiny_imagenet.py first.")
        
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

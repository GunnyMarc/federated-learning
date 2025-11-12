"""
Data utilities for federated learning.

This module provides functions for loading datasets and creating federated data distributions,
including both IID (Independent and Identically Distributed) and non-IID scenarios.
"""

from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np


def load_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Load a standard dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST')
        data_dir (str): Directory to store/load the dataset
        
    Returns:
        Tuple[Dataset, Dataset]: Training and test datasets
    """
    if dataset_name.upper() == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        
    elif dataset_name.upper() == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.upper() == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.upper() == 'FASHIONMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def create_iid_split(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    Create IID (Independent and Identically Distributed) split of the dataset.
    
    Each client receives an equal number of samples with uniform distribution of classes.
    
    Args:
        dataset (Dataset): The dataset to split
        num_clients (int): Number of clients
        
    Returns:
        List[Subset]: List of dataset subsets, one per client
    """
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    # Split indices among clients
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def create_non_iid_split(
    dataset: Dataset,
    num_clients: int,
    classes_per_client: int = 2,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Create non-IID split of the dataset using Dirichlet distribution.
    
    This creates a more realistic scenario where each client has a skewed distribution
    of classes, simulating real-world personalized data.
    
    Args:
        dataset (Dataset): The dataset to split
        num_clients (int): Number of clients
        classes_per_client (int): Number of classes each client should have
        alpha (float): Dirichlet distribution parameter (lower = more skewed)
        
    Returns:
        List[Subset]: List of dataset subsets, one per client
    """
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Try to extract labels from the dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Method 1: Simple assignment of specific classes per client
    if classes_per_client > 0:
        client_datasets = []
        
        for client_id in range(num_clients):
            # Randomly select classes for this client
            client_classes = np.random.choice(
                num_classes, 
                size=min(classes_per_client, num_classes), 
                replace=False
            )
            
            # Get indices for selected classes
            client_indices = []
            for cls in client_classes:
                class_indices = np.where(labels == cls)[0]
                # Take a portion of samples from this class
                num_samples = len(class_indices) // (num_clients // classes_per_client + 1)
                selected = np.random.choice(class_indices, size=num_samples, replace=False)
                client_indices.extend(selected)
            
            client_datasets.append(Subset(dataset, client_indices))
    
    # Method 2: Dirichlet distribution (more sophisticated)
    else:
        # Use Dirichlet distribution to assign samples
        min_size = 0
        N = len(labels)
        
        client_indices = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_clients) 
                                   for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            client_indices = [idx_j + idx.tolist() 
                            for idx_j, idx in zip(client_indices, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        
        client_datasets = [Subset(dataset, idx) for idx in client_indices]
    
    return client_datasets


def create_federated_dataloaders(
    num_clients: int,
    dataset_name: str = 'CIFAR10',
    batch_size: int = 32,
    iid: bool = True,
    classes_per_client: int = 2,
    alpha: float = 0.5,
    data_dir: str = './data',
    num_workers: int = 2
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create federated data loaders for training and a central test loader.
    
    Args:
        num_clients (int): Number of clients
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size for training
        iid (bool): Whether to use IID distribution
        classes_per_client (int): Number of classes per client (for non-IID)
        alpha (float): Dirichlet parameter (for non-IID)
        data_dir (str): Directory to store data
        num_workers (int): Number of workers for data loading
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: List of client train loaders and test loader
    """
    # Load datasets
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)
    
    # Create federated split
    if iid:
        client_datasets = create_iid_split(train_dataset, num_clients)
    else:
        client_datasets = create_non_iid_split(
            train_dataset, num_clients, classes_per_client, alpha
        )
    
    # Create data loaders
    train_loaders = [
        DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        for client_dataset in client_datasets
    ]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loaders, test_loader


def create_common_dataset(
    dataset_name: str = 'CIFAR10',
    num_samples: int = 1000,
    data_dir: str = './data'
) -> DataLoader:
    """
    Create a small common dataset for fine-tuning in non-IID scenarios.
    
    Args:
        dataset_name (str): Name of the dataset
        num_samples (int): Number of samples in the common dataset
        data_dir (str): Directory to store data
        
    Returns:
        DataLoader: DataLoader for the common dataset
    """
    train_dataset, _ = load_dataset(dataset_name, data_dir)
    
    # Randomly sample from training set
    indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
    common_dataset = Subset(train_dataset, indices)
    
    common_loader = DataLoader(
        common_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    return common_loader


def analyze_data_distribution(
    client_datasets: List[Subset],
    num_classes: int = 10
) -> np.ndarray:
    """
    Analyze the class distribution across clients.
    
    Args:
        client_datasets (List[Subset]): List of client datasets
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Matrix of shape (num_clients, num_classes) with class counts
    """
    num_clients = len(client_datasets)
    distribution = np.zeros((num_clients, num_classes))
    
    for client_id, client_dataset in enumerate(client_datasets):
        for idx in client_dataset.indices:
            label = client_dataset.dataset[idx][1]
            distribution[client_id, label] += 1
    
    return distribution


if __name__ == "__main__":
    print("Testing data utilities...")
    
    # Test IID split
    print("\n1. Testing IID split...")
    train_loaders, test_loader = create_federated_dataloaders(
        num_clients=8,
        dataset_name='CIFAR10',
        iid=True
    )
    print(f"Number of client loaders: {len(train_loaders)}")
    print(f"Samples per client: {[len(loader.dataset) for loader in train_loaders]}")
    
    # Test non-IID split
    print("\n2. Testing non-IID split...")
    train_loaders_non_iid, _ = create_federated_dataloaders(
        num_clients=8,
        dataset_name='CIFAR10',
        iid=False,
        classes_per_client=3
    )
    print(f"Number of client loaders: {len(train_loaders_non_iid)}")
    print(f"Samples per client: {[len(loader.dataset) for loader in train_loaders_non_iid]}")
    
    # Analyze distribution
    print("\n3. Analyzing data distribution...")
    client_datasets = [loader.dataset for loader in train_loaders_non_iid]
    distribution = analyze_data_distribution(client_datasets)
    print(f"Distribution matrix shape: {distribution.shape}")
    print(f"Client 0 class distribution: {distribution[0]}")
    
    print("\nAll tests passed!")

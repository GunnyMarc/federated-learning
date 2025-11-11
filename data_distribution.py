"""
Data Distribution Module for Federated Learning

Handles creation of IID and Non-IID data distributions across clients.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class DataDistributor:
    """
    Handles distribution of data across federated learning clients.
    """

    def __init__(self, dataset_name='cifar10', data_dir='./data'):
        """
        Initialize the data distributor.

        Args:
            dataset_name (str): Name of the dataset ('cifar10', 'mnist')
            data_dir (str): Directory to store/load data
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Get data transformations."""
        if self.dataset_name == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        elif self.dataset_name == 'mnist':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.ToTensor()

    def load_dataset(self, train=True):
        """
        Load the dataset.

        Args:
            train (bool): Whether to load training or test data

        Returns:
            Dataset: PyTorch dataset
        """
        if self.dataset_name == 'cifar10':
            return datasets.CIFAR10(
                root=self.data_dir,
                train=train,
                download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'mnist':
            return datasets.MNIST(
                root=self.data_dir,
                train=train,
                download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def create_iid_distribution(
            self,
            num_clients: int,
            batch_size: int = 32
    ) -> Tuple[List[DataLoader], DataLoader]:
        """
        Create IID (Independent and Identically Distributed) data distribution.

        Each client gets an equal share of uniformly distributed data.

        Args:
            num_clients (int): Number of clients
            batch_size (int): Batch size for DataLoader

        Returns:
            tuple: (list of client DataLoaders, test DataLoader)
        """
        # Load training data
        train_dataset = self.load_dataset(train=True)
        test_dataset = self.load_dataset(train=False)

        # Calculate samples per client
        total_samples = len(train_dataset)
        samples_per_client = total_samples // num_clients

        # Shuffle indices
        indices = np.random.permutation(total_samples)

        # Create client dataloaders
        client_loaders = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples

            client_indices = indices[start_idx:end_idx]
            client_dataset = Subset(train_dataset, client_indices)

            client_loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            client_loaders.append(client_loader)

        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return client_loaders, test_loader

    def create_non_iid_distribution(
            self,
            num_clients: int,
            batch_size: int = 32,
            shards_per_client: int = 2,
            alpha: float = 0.5
    ) -> Tuple[List[DataLoader], DataLoader]:
        """
        Create Non-IID data distribution using Dirichlet distribution.

        Clients have heterogeneous data distributions that don't match
        the overall population distribution.

        Args:
            num_clients (int): Number of clients
            batch_size (int): Batch size for DataLoader
            shards_per_client (int): Number of class shards per client
            alpha (float): Dirichlet concentration parameter (lower = more skewed)

        Returns:
            tuple: (list of client DataLoaders, test DataLoader)
        """
        # Load datasets
        train_dataset = self.load_dataset(train=True)
        test_dataset = self.load_dataset(train=False)

        # Get labels
        if hasattr(train_dataset, 'targets'):
            labels = np.array(train_dataset.targets)
        else:
            labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

        num_classes = len(np.unique(labels))

        # Sort indices by label
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # Shuffle within each class
        for indices in label_indices:
            np.random.shuffle(indices)

        # Distribute using Dirichlet distribution
        client_indices_list = [[] for _ in range(num_clients)]

        for class_indices in label_indices:
            # Generate proportions using Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)

            # Adjust to ensure all samples are distributed
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            # Distribute indices
            start_idx = 0
            for client_id, num_samples in enumerate(proportions):
                end_idx = start_idx + num_samples
                client_indices_list[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

        # Create client dataloaders
        client_loaders = []
        for client_indices in client_indices_list:
            np.random.shuffle(client_indices)
            client_dataset = Subset(train_dataset, client_indices)

            client_loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            client_loaders.append(client_loader)

        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return client_loaders, test_loader

    def create_pathological_non_iid_distribution(
            self,
            num_clients: int,
            batch_size: int = 32,
            classes_per_client: int = 2
    ) -> Tuple[List[DataLoader], DataLoader]:
        """
        Create pathological Non-IID distribution.

        Each client only has data from a limited number of classes.

        Args:
            num_clients (int): Number of clients
            batch_size (int): Batch size for DataLoader
            classes_per_client (int): Number of classes each client has

        Returns:
            tuple: (list of client DataLoaders, test DataLoader)
        """
        # Load datasets
        train_dataset = self.load_dataset(train=True)
        test_dataset = self.load_dataset(train=False)

        # Get labels
        if hasattr(train_dataset, 'targets'):
            labels = np.array(train_dataset.targets)
        else:
            labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

        num_classes = len(np.unique(labels))

        # Sort indices by label
        label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

        # Shuffle within each class
        for indices in label_indices.values():
            np.random.shuffle(indices)

        # Assign classes to clients
        client_classes = []
        available_classes = list(range(num_classes))

        for _ in range(num_clients):
            if len(available_classes) < classes_per_client:
                available_classes = list(range(num_classes))
                np.random.shuffle(available_classes)

            client_class_subset = available_classes[:classes_per_client]
            available_classes = available_classes[classes_per_client:]
            client_classes.append(client_class_subset)

        # Distribute data
        client_loaders = []
        for client_class_subset in client_classes:
            client_indices = []

            for class_id in client_class_subset:
                class_samples = label_indices[class_id]
                samples_per_class = len(class_samples) // (num_clients // classes_per_client + 1)

                client_indices.extend(class_samples[:samples_per_class].tolist())
                label_indices[class_id] = class_samples[samples_per_class:]

            np.random.shuffle(client_indices)
            client_dataset = Subset(train_dataset, client_indices)

            client_loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            client_loaders.append(client_loader)

        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return client_loaders, test_loader

    def visualize_distribution(self, client_loaders: List[DataLoader], num_classes: int = 10):
        """
        Visualize the data distribution across clients.

        Args:
            client_loaders (list): List of client DataLoaders
            num_classes (int): Number of classes in the dataset
        """
        num_clients = len(client_loaders)

        # Count samples per class for each client
        client_distributions = np.zeros((num_clients, num_classes))

        for client_id, loader in enumerate(client_loaders):
            for _, labels in loader:
                for label in labels:
                    client_distributions[client_id, label.item()] += 1

        # Create visualization
        fig, axes = plt.subplots(
            (num_clients + 3) // 4, 4,
            figsize=(16, 4 * ((num_clients + 3) // 4))
        )
        axes = axes.flatten()

        for client_id in range(num_clients):
            ax = axes[client_id]
            ax.bar(range(num_classes), client_distributions[client_id])
            ax.set_title(f'Client {client_id}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_xticks(range(num_classes))

        # Hide unused subplots
        for idx in range(num_clients, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('client_data_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

        return client_distributions


def create_common_dataset(
        dataset_name='cifar10',
        data_dir='./data',
        num_samples=500,
        batch_size=32
) -> DataLoader:
    """
    Create a small common dataset for fine-tuning in Non-IID scenarios.

    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Directory to store/load data
        num_samples (int): Number of samples in common dataset
        batch_size (int): Batch size

    Returns:
        DataLoader: Common dataset loader
    """
    distributor = DataDistributor(dataset_name, data_dir)
    train_dataset = distributor.load_dataset(train=True)

    # Sample balanced subset
    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    num_classes = len(np.unique(labels))
    samples_per_class = num_samples // num_classes

    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
        selected_indices.extend(selected.tolist())

    common_dataset = Subset(train_dataset, selected_indices)
    common_loader = DataLoader(common_dataset, batch_size=batch_size, shuffle=True)

    return common_loader


if __name__ == "__main__":
    # Test data distribution
    print("Testing IID Distribution...")
    distributor = DataDistributor('cifar10')
    client_loaders, test_loader = distributor.create_iid_distribution(num_clients=8)

    print(f"Number of clients: {len(client_loaders)}")
    print(f"Samples in first client: {len(client_loaders[0].dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("\nVisualizing IID distribution...")
    distributor.visualize_distribution(client_loaders)

    print("\nTesting Non-IID Distribution...")
    client_loaders, test_loader = distributor.create_non_iid_distribution(
        num_clients=8,
        alpha=0.5
    )

    print("\nVisualizing Non-IID distribution...")
    distributor.visualize_distribution(client_loaders)
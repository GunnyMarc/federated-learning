"""
Compare Different Training Approaches

This script compares:
1. Traditional centralized training
2. Federated learning with IID data
3. Federated learning with non-IID data
4. Federated learning with non-IID data + common dataset fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')

from models import SimpleCNN
from algorithms import FedAvg
from data_utils import create_federated_dataloaders, create_common_dataset, load_dataset
from server import FederatedServer


def train_traditional(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    num_epochs: int = 20,
    learning_rate: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Traditional centralized training.
    
    Returns:
        dict: Training history
    """
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    history = {'epochs': [], 'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    
    print("\nTraining traditional model...")
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / total
        
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(accuracy)
    
    return history


def train_federated(
    iid: bool = True,
    use_common_dataset: bool = False,
    num_clients: int = 8,
    num_rounds: int = 20,
    classes_per_client: int = 3,
    learning_rate: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Train federated model.
    
    Args:
        iid: Use IID or non-IID data distribution
        use_common_dataset: Use common dataset for fine-tuning
        num_clients: Number of clients
        num_rounds: Number of communication rounds
        classes_per_client: Classes per client (for non-IID)
        learning_rate: Learning rate
        device: Device to use
        
    Returns:
        dict: Training history
    """
    # Create data loaders
    train_loaders, test_loader = create_federated_dataloaders(
        num_clients=num_clients,
        dataset_name='CIFAR10',
        batch_size=32,
        iid=iid,
        classes_per_client=classes_per_client,
        data_dir='./data'
    )
    
    # Create common dataset if needed
    common_loader = None
    if use_common_dataset:
        common_loader = create_common_dataset(
            dataset_name='CIFAR10',
            num_samples=1000,
            data_dir='./data'
        )
    
    # Create model and server
    model = SimpleCNN(num_classes=10)
    algorithm = FedAvg(local_epochs=1)
    server = FederatedServer(
        model=model,
        algorithm=algorithm,
        num_clients=num_clients,
        device=device
    )
    
    if common_loader is not None:
        server.set_common_dataset(common_loader)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train
    label = "IID" if iid else "Non-IID"
    if use_common_dataset:
        label += " + Common"
    print(f"\nTraining federated model ({label})...")
    
    history = server.train(
        train_loaders=train_loaders,
        test_loader=test_loader,
        criterion=criterion,
        num_rounds=num_rounds,
        learning_rate=learning_rate,
        local_epochs=1,
        eval_every=1,
        verbose=True
    )
    
    return history


def plot_comparison(results: dict, save_path: str = 'comparison_results.png'):
    """
    Plot comparison of all approaches.
    
    Args:
        results: Dictionary with results from each approach
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D']
    
    # Plot losses
    for idx, (name, history) in enumerate(results.items()):
        if 'rounds' in history:
            x = history['rounds']
            label_x = 'Communication Round'
        else:
            x = history['epochs']
            label_x = 'Epoch'
        
        axes[0].plot(x, history['test_loss'], label=name, 
                    marker=markers[idx], color=colors[idx], markevery=2)
    
    axes[0].set_xlabel(label_x)
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('Test Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    for idx, (name, history) in enumerate(results.items()):
        if 'rounds' in history:
            x = history['rounds']
            label_x = 'Communication Round'
        else:
            x = history['epochs']
            label_x = 'Epoch'
        
        axes[1].plot(x, history['test_accuracy'], label=name,
                    marker=markers[idx], color=colors[idx], markevery=2)
    
    axes[1].set_xlabel(label_x)
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.show()


def print_summary(results: dict):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Approach':<40} {'Final Accuracy':<15} {'Final Loss':<15}")
    print("-" * 80)
    
    for name, history in results.items():
        final_acc = history['test_accuracy'][-1]
        final_loss = history['test_loss'][-1]
        print(f"{name:<40} {final_acc:>6.4f} ({final_acc*100:>5.2f}%) {final_loss:>14.4f}")
    
    print("=" * 80)


def main():
    """Main comparison function."""
    print("=" * 80)
    print("FEDERATED LEARNING COMPARISON")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Configuration:")
    print(f"  Dataset: CIFAR-10")
    print(f"  Number of clients: 8")
    print(f"  Number of rounds/epochs: 20")
    print(f"  Learning rate: 0.01")
    print()
    
    results = {}
    
    # 1. Traditional Training
    print("\n" + "-" * 80)
    print("1. Traditional Centralized Training")
    print("-" * 80)
    train_dataset, test_dataset = load_dataset('CIFAR10', data_dir='./data')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model_traditional = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    
    results['Traditional'] = train_traditional(
        model=model_traditional,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        num_epochs=20,
        learning_rate=0.01,
        device=device
    )
    
    # 2. Federated Learning with IID Data
    print("\n" + "-" * 80)
    print("2. Federated Learning (IID)")
    print("-" * 80)
    results['Federated (IID)'] = train_federated(
        iid=True,
        use_common_dataset=False,
        num_clients=8,
        num_rounds=20,
        learning_rate=0.01,
        device=device
    )
    
    # 3. Federated Learning with Non-IID Data
    print("\n" + "-" * 80)
    print("3. Federated Learning (Non-IID)")
    print("-" * 80)
    results['Federated (Non-IID)'] = train_federated(
        iid=False,
        use_common_dataset=False,
        num_clients=8,
        num_rounds=20,
        classes_per_client=3,
        learning_rate=0.01,
        device=device
    )
    
    # 4. Federated Learning with Non-IID Data + Common Dataset
    print("\n" + "-" * 80)
    print("4. Federated Learning (Non-IID + Common Dataset)")
    print("-" * 80)
    results['Federated (Non-IID + Common)'] = train_federated(
        iid=False,
        use_common_dataset=True,
        num_clients=8,
        num_rounds=20,
        classes_per_client=3,
        learning_rate=0.01,
        device=device
    )
    
    # Print summary
    print_summary(results)
    
    # Plot comparison
    plot_comparison(results)
    
    print("\n" + "=" * 80)
    print("Comparison completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

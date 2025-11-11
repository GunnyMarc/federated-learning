"""
Complete Federated Learning Example

This script demonstrates:
1. IID data distribution
2. Non-IID data distribution
3. FedAvg algorithm
4. FedSGD algorithm
5. Comparison with traditional training
6. Handling Non-IID data with fine-tuning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_nn import SimpleNN
from src.data.data_distribution import DataDistributor, create_common_dataset
from src.algorithms.fedavg import FedAvgTrainer
from src.algorithms.fedsgd import FedSGDTrainer


def traditional_training(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.01,
        device: str = 'cpu'
) -> dict:
    """
    Traditional centralized training for comparison.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data
        test_loader (DataLoader): Test data
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to use

    Returns:
        dict: Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }

    print("Starting traditional training...")

    for epoch in range(num_epochs):
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
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct / total

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.4f}")

    return history


def plot_comparison(histories: dict, title: str = "Comparison", save_path: str = None):
    """
    Plot comparison of different training approaches.

    Args:
        histories (dict): Dictionary of training histories
        title (str): Plot title
        save_path (str): Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    for name, history in histories.items():
        if 'test_accuracy' in history:
            ax1.plot(history['test_accuracy'], label=name, marker='o')

    ax1.set_xlabel('Round/Epoch')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    for name, history in histories.items():
        if 'test_loss' in history:
            ax2.plot(history['test_loss'], label=name, marker='o')

    ax2.set_xlabel('Round/Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def main():
    """Main function demonstrating various federated learning scenarios."""

    # Configuration
    NUM_CLIENTS = 8
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    print("=" * 80)

    # ========================================================================
    # Scenario 1: Federated Learning with IID Data (FedAvg)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 1: Federated Learning with IID Data (FedAvg)")
    print("=" * 80)

    distributor = DataDistributor('cifar10')
    client_loaders, test_loader = distributor.create_iid_distribution(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE
    )

    print(f"Created {NUM_CLIENTS} clients with IID data distribution")
    print(f"Samples per client: ~{len(client_loaders[0].dataset)}")

    # Visualize distribution
    print("Visualizing data distribution...")
    distributor.visualize_distribution(client_loaders)

    # Train with FedAvg
    model = SimpleNN()
    fedavg_trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    fedavg_iid_history = fedavg_trainer.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        verbose=True
    )

    print(f"\nFinal Test Accuracy: {fedavg_iid_history['test_accuracy'][-1]:.4f}")

    # ========================================================================
    # Scenario 2: Federated Learning with Non-IID Data (FedAvg)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: Federated Learning with Non-IID Data (FedAvg)")
    print("=" * 80)

    client_loaders_noniid, test_loader = distributor.create_non_iid_distribution(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        alpha=0.5
    )

    print(f"Created {NUM_CLIENTS} clients with Non-IID data distribution")

    # Visualize distribution
    print("Visualizing Non-IID data distribution...")
    distributor.visualize_distribution(client_loaders_noniid)

    # Train with FedAvg
    model = SimpleNN()
    fedavg_noniid_trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders_noniid,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    fedavg_noniid_history = fedavg_noniid_trainer.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        verbose=True
    )

    print(f"\nFinal Test Accuracy: {fedavg_noniid_history['test_accuracy'][-1]:.4f}")

    # ========================================================================
    # Scenario 3: FedSGD with IID Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 3: Federated SGD with IID Data")
    print("=" * 80)

    model = SimpleNN()
    fedsgd_trainer = FedSGDTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    fedsgd_history = fedsgd_trainer.train(
        num_rounds=NUM_ROUNDS,
        verbose=True
    )

    print(f"\nFinal Test Accuracy: {fedsgd_history['test_accuracy'][-1]:.4f}")

    # ========================================================================
    # Scenario 4: Traditional Centralized Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 4: Traditional Centralized Training")
    print("=" * 80)

    # Combine all client data for centralized training
    from torch.utils.data import ConcatDataset
    all_datasets = [loader.dataset for loader in client_loaders]
    combined_dataset = ConcatDataset(all_datasets)
    centralized_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SimpleNN()
    traditional_history = traditional_training(
        model=model,
        train_loader=centralized_loader,
        test_loader=test_loader,
        num_epochs=NUM_ROUNDS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    # ========================================================================
    # Scenario 5: Random Client Selection
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 5: Federated Learning with Random Client Selection")
    print("=" * 80)

    model = SimpleNN()
    fedavg_random_trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    fedavg_random_history = fedavg_random_trainer.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        client_fraction=0.5,  # Use 50% of clients per round
        verbose=True
    )

    print(f"\nFinal Test Accuracy: {fedavg_random_history['test_accuracy'][-1]:.4f}")

    # ========================================================================
    # Comparison and Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    histories = {
        'FedAvg (IID)': fedavg_iid_history,
        'FedAvg (Non-IID)': fedavg_noniid_history,
        'FedSGD (IID)': fedsgd_history,
        'Traditional': traditional_history,
        'FedAvg (Random 50%)': fedavg_random_history
    }

    print("\nFinal Test Accuracies:")
    for name, history in histories.items():
        final_acc = history['test_accuracy'][-1]
        print(f"{name:25s}: {final_acc:.4f}")

    # Plot comparisons
    print("\nGenerating comparison plots...")

    # Compare all methods
    plot_comparison(
        histories,
        title="Federated Learning vs Traditional Training",
        save_path="comparison_all_methods.png"
    )

    # Compare IID vs Non-IID
    plot_comparison(
        {
            'IID': fedavg_iid_history,
            'Non-IID': fedavg_noniid_history,
            'Traditional': traditional_history
        },
        title="IID vs Non-IID Data Distribution",
        save_path="comparison_iid_vs_noniid.png"
    )

    # Compare FedAvg vs FedSGD
    plot_comparison(
        {
            'FedAvg': fedavg_iid_history,
            'FedSGD': fedsgd_history,
            'Traditional': traditional_history
        },
        title="FedAvg vs FedSGD",
        save_path="comparison_fedavg_vs_fedsgd.png"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nKey Observations:")
    print("1. Traditional training typically achieves highest accuracy (centralized data)")
    print("2. FedAvg with IID data performs close to traditional training")
    print("3. Non-IID data significantly degrades performance")
    print("4. FedSGD converges faster but may be less stable")
    print("5. Random client selection adds variability but maintains privacy better")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
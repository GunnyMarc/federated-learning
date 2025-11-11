"""
Handling Non-IID Data in Federated Learning

This example demonstrates:
1. The challenge of Non-IID data
2. Fine-tuning approach to mitigate Non-IID issues
3. Comparison of results
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_nn import SimpleNN
from src.data.data_distribution import DataDistributor, create_common_dataset
from src.algorithms.fedavg import FedAvgClient, FedAvgServer
from tqdm import tqdm
import copy


class FedAvgWithFineTuning:
    """
    FedAvg trainer with fine-tuning on common dataset to handle Non-IID data.
    """

    def __init__(
            self,
            model: nn.Module,
            client_loaders: list,
            test_loader: DataLoader,
            common_loader: DataLoader = None,
            learning_rate: float = 0.01,
            device: str = 'cpu'
    ):
        self.server = FedAvgServer(copy.deepcopy(model), device)
        self.clients = [
            FedAvgClient(copy.deepcopy(model), loader, learning_rate, device)
            for loader in client_loaders
        ]
        self.test_loader = test_loader
        self.common_loader = common_loader
        self.device = device

        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }

    def fine_tune_client_models(
            self,
            client_models: list,
            epochs: int = 1
    ):
        """
        Fine-tune client models on common dataset.

        Args:
            client_models (list): List of client model weights
            epochs (int): Number of fine-tuning epochs

        Returns:
            list: Fine-tuned model weights
        """
        if self.common_loader is None:
            return client_models

        fine_tuned_models = []
        criterion = nn.CrossEntropyLoss()

        for weights in client_models:
            # Create temporary model
            temp_model = copy.deepcopy(self.clients[0].model)
            temp_model.to(self.device)

            # Set weights
            with torch.no_grad():
                for param, weight in zip(temp_model.parameters(), weights):
                    param.data.copy_(weight.to(self.device))

            # Fine-tune
            optimizer = optim.SGD(temp_model.parameters(), lr=0.01, momentum=0.9)
            temp_model.train()

            for _ in range(epochs):
                for data, target in self.common_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = temp_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # Get fine-tuned weights
            fine_tuned_weights = [
                param.data.clone().cpu() for param in temp_model.parameters()
            ]
            fine_tuned_models.append(fine_tuned_weights)

        return fine_tuned_models

    def train_round(
            self,
            local_epochs: int = 1,
            fine_tune: bool = False
    ):
        """Execute one training round with optional fine-tuning."""
        # Get global weights
        global_weights = self.server.get_weights()

        # Collect updates from clients
        client_weights = []
        client_sizes = []
        client_losses = []

        for client in self.clients:
            client.set_weights(global_weights)
            weights, loss = client.train(epochs=local_epochs)

            client_weights.append(weights)
            client_sizes.append(client.get_dataset_size())
            client_losses.append(loss)

        # Fine-tune if enabled
        if fine_tune and self.common_loader is not None:
            client_weights = self.fine_tune_client_models(client_weights)

        # Aggregate
        aggregated_weights = self.server.aggregate_weights(
            client_weights,
            client_sizes
        )

        # Update global model
        self.server.set_weights(aggregated_weights)

        # Evaluate
        test_acc, test_loss = self.server.evaluate(self.test_loader)

        return {
            'avg_train_loss': sum(client_losses) / len(client_losses),
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

    def train(
            self,
            num_rounds: int,
            local_epochs: int = 1,
            fine_tune: bool = False,
            verbose: bool = True
    ):
        """Train the federated model."""
        iterator = tqdm(range(num_rounds)) if verbose else range(num_rounds)

        for round_num in iterator:
            stats = self.train_round(local_epochs, fine_tune)

            self.history['train_loss'].append(stats['avg_train_loss'])
            self.history['test_loss'].append(stats['test_loss'])
            self.history['test_accuracy'].append(stats['test_accuracy'])

            if verbose:
                fine_tune_str = " (with fine-tuning)" if fine_tune else ""
                iterator.set_description(
                    f"Round {round_num + 1}/{num_rounds}{fine_tune_str} | "
                    f"Loss: {stats['test_loss']:.4f} | "
                    f"Acc: {stats['test_accuracy']:.4f}"
                )

        return self.history


def main():
    """Main function demonstrating Non-IID handling."""

    print("=" * 80)
    print("HANDLING NON-IID DATA IN FEDERATED LEARNING")
    print("=" * 80)

    # Configuration
    NUM_CLIENTS = 8
    NUM_ROUNDS = 10
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    COMMON_DATASET_SIZE = 500

    print(f"\nConfiguration:")
    print(f"  - Number of clients: {NUM_CLIENTS}")
    print(f"  - Training rounds: {NUM_ROUNDS}")
    print(f"  - Common dataset size: {COMMON_DATASET_SIZE}")
    print(f"  - Device: {DEVICE}")

    # ========================================================================
    # Setup 1: Non-IID Data Without Fine-tuning
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Non-IID Data WITHOUT Fine-tuning")
    print("=" * 80)

    distributor = DataDistributor('cifar10')
    client_loaders, test_loader = distributor.create_non_iid_distribution(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        alpha=0.5
    )

    print("\nVisualizing Non-IID data distribution...")
    distributor.visualize_distribution(client_loaders)

    # Train without fine-tuning
    model = SimpleNN()
    trainer_no_finetune = FedAvgWithFineTuning(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        common_loader=None,
        learning_rate=0.01,
        device=DEVICE
    )

    history_no_finetune = trainer_no_finetune.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=1,
        fine_tune=False,
        verbose=True
    )

    # ========================================================================
    # Setup 2: Non-IID Data With Fine-tuning
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Non-IID Data WITH Fine-tuning")
    print("=" * 80)

    # Create common dataset
    print(f"\nCreating common dataset with {COMMON_DATASET_SIZE} samples...")
    common_loader = create_common_dataset(
        dataset_name='cifar10',
        num_samples=COMMON_DATASET_SIZE,
        batch_size=BATCH_SIZE
    )

    print(f"✓ Common dataset created")

    # Train with fine-tuning
    model = SimpleNN()
    trainer_finetune = FedAvgWithFineTuning(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        common_loader=common_loader,
        learning_rate=0.01,
        device=DEVICE
    )

    history_finetune = trainer_finetune.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=1,
        fine_tune=True,
        verbose=True
    )

    # ========================================================================
    # Setup 3: IID Data for Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: IID Data for Comparison")
    print("=" * 80)

    client_loaders_iid, test_loader = distributor.create_iid_distribution(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE
    )

    model = SimpleNN()
    trainer_iid = FedAvgWithFineTuning(
        model=model,
        client_loaders=client_loaders_iid,
        test_loader=test_loader,
        common_loader=None,
        learning_rate=0.01,
        device=DEVICE
    )

    history_iid = trainer_iid.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=1,
        fine_tune=False,
        verbose=True
    )

    # ========================================================================
    # Results and Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\nFinal Test Accuracies:")
    print(f"  IID Data:                    {history_iid['test_accuracy'][-1]:.4f}")
    print(f"  Non-IID (No Fine-tuning):    {history_no_finetune['test_accuracy'][-1]:.4f}")
    print(f"  Non-IID (With Fine-tuning):  {history_finetune['test_accuracy'][-1]:.4f}")

    improvement = (history_finetune['test_accuracy'][-1] -
                   history_no_finetune['test_accuracy'][-1]) * 100
    print(f"\nImprovement from Fine-tuning: {improvement:.2f}%")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history_iid['test_accuracy'],
             label='IID', marker='o', linewidth=2)
    ax1.plot(history_no_finetune['test_accuracy'],
             label='Non-IID (No Fine-tuning)', marker='s', linewidth=2)
    ax1.plot(history_finetune['test_accuracy'],
             label='Non-IID (With Fine-tuning)', marker='^', linewidth=2)

    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy: Impact of Fine-tuning', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history_iid['test_loss'],
             label='IID', marker='o', linewidth=2)
    ax2.plot(history_no_finetune['test_loss'],
             label='Non-IID (No Fine-tuning)', marker='s', linewidth=2)
    ax2.plot(history_finetune['test_loss'],
             label='Non-IID (With Fine-tuning)', marker='^', linewidth=2)

    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test Loss: Impact of Fine-tuning', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noniid_finetuning_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("\n1. Non-IID data significantly reduces model performance")
    print("2. Fine-tuning on a small common dataset helps mitigate this issue")
    print(f"3. With just {COMMON_DATASET_SIZE} common samples, we achieve ~{improvement:.1f}% improvement")
    print("4. IID data still performs best, but fine-tuning closes the gap")
    print("\nConclusion: Fine-tuning is an effective strategy for handling Non-IID data!")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    main()
"""
Complete End-to-End Federated Learning Example

This script demonstrates a complete federated learning workflow from data preparation
to training, evaluation, and visualization.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from models import SimpleCNN
from algorithms import FedAvg
from data_utils import (
    create_federated_dataloaders, 
    create_common_dataset,
    analyze_data_distribution
)
from server import FederatedServer


def visualize_data_distribution(train_loaders, title="Data Distribution"):
    """Visualize how data is distributed across clients."""
    client_datasets = [loader.dataset for loader in train_loaders]
    distribution = analyze_data_distribution(client_datasets, num_classes=10)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    
    for i in range(min(8, len(train_loaders))):
        axes[i].bar(range(10), distribution[i])
        axes[i].set_title(f'Client {i+1}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Samples')
        axes[i].set_ylim([0, distribution.max() * 1.1])
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history, title="Training Progress"):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['rounds'], history['train_loss'], 
                label='Train Loss', marker='o', color='blue')
    axes[0].plot(history['rounds'], history['test_loss'], 
                label='Test Loss', marker='s', color='red')
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['rounds'], 
                [acc * 100 for acc in history['test_accuracy']], 
                marker='o', color='green', linewidth=2)
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Accuracy improvement
    if len(history['test_accuracy']) > 1:
        improvements = [0] + [
            (history['test_accuracy'][i] - history['test_accuracy'][i-1]) * 100
            for i in range(1, len(history['test_accuracy']))
        ]
        axes[2].bar(history['rounds'], improvements, color='purple', alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[2].set_xlabel('Communication Round')
        axes[2].set_ylabel('Accuracy Improvement (%)')
        axes[2].set_title('Round-to-Round Improvement')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_experiment(
    experiment_name: str,
    iid: bool = True,
    use_common_dataset: bool = False,
    num_clients: int = 8,
    num_rounds: int = 15,
    classes_per_client: int = 3
):
    """Run a complete federated learning experiment."""
    
    print("\n" + "=" * 80)
    print(f"Experiment: {experiment_name}")
    print("=" * 80)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Data distribution: {'IID' if iid else 'Non-IID'}")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    if not iid:
        print(f"  Classes per client: {classes_per_client}")
    print(f"  Use common dataset: {use_common_dataset}")
    print()
    
    # Create data loaders
    print("Creating federated data loaders...")
    train_loaders, test_loader = create_federated_dataloaders(
        num_clients=num_clients,
        dataset_name='CIFAR10',
        batch_size=32,
        iid=iid,
        classes_per_client=classes_per_client,
        data_dir='./data'
    )
    print(f"✓ Created {len(train_loaders)} client data loaders")
    print(f"✓ Average samples per client: {np.mean([len(l.dataset) for l in train_loaders]):.0f}")
    
    # Visualize data distribution
    print("\nVisualizing data distribution...")
    visualize_data_distribution(
        train_loaders, 
        title=f"{experiment_name} - Data Distribution"
    )
    
    # Create common dataset if needed
    common_loader = None
    if use_common_dataset:
        print("\nCreating common dataset for fine-tuning...")
        common_loader = create_common_dataset(
            dataset_name='CIFAR10',
            num_samples=1000,
            data_dir='./data'
        )
        print(f"✓ Common dataset size: {len(common_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: SimpleCNN")
    print(f"✓ Total parameters: {num_params:,}")
    
    # Create server
    print("\nInitializing federated server...")
    algorithm = FedAvg(local_epochs=1)
    server = FederatedServer(
        model=model,
        algorithm=algorithm,
        num_clients=num_clients
    )
    
    if common_loader is not None:
        server.set_common_dataset(common_loader)
        print("✓ Common dataset configured")
    
    # Train
    print("\nStarting federated training...")
    print("-" * 80)
    
    criterion = nn.CrossEntropyLoss()
    history = server.train(
        train_loaders=train_loaders,
        test_loader=test_loader,
        criterion=criterion,
        num_rounds=num_rounds,
        learning_rate=0.01,
        local_epochs=1,
        eval_every=1,
        verbose=True
    )
    
    print("-" * 80)
    
    # Results
    final_accuracy = history['test_accuracy'][-1]
    best_accuracy = max(history['test_accuracy'])
    
    print("\nResults:")
    print(f"  Final test accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  Best test accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"  Final test loss:     {history['test_loss'][-1]:.4f}")
    
    # Plot results
    print("\nGenerating training curves...")
    plot_training_curves(history, title=f"{experiment_name} - Training Progress")
    
    # Save model
    model_path = f"{experiment_name.lower().replace(' ', '_')}_model.pth"
    server.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print(f"Experiment '{experiment_name}' completed!")
    print("=" * 80)
    
    return history


def main():
    """Run multiple experiments and compare results."""
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING: COMPLETE DEMONSTRATION")
    print("=" * 80)
    print("\nThis script will run three experiments:")
    print("  1. IID data distribution")
    print("  2. Non-IID data distribution")
    print("  3. Non-IID with common dataset fine-tuning")
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Experiment 1: IID
    history_iid = run_experiment(
        experiment_name="IID Distribution",
        iid=True,
        use_common_dataset=False,
        num_clients=8,
        num_rounds=15
    )
    
    # Experiment 2: Non-IID
    history_non_iid = run_experiment(
        experiment_name="Non-IID Distribution",
        iid=False,
        use_common_dataset=False,
        num_clients=8,
        num_rounds=15,
        classes_per_client=3
    )
    
    # Experiment 3: Non-IID with common dataset
    history_non_iid_common = run_experiment(
        experiment_name="Non-IID with Common Dataset",
        iid=False,
        use_common_dataset=True,
        num_clients=8,
        num_rounds=15,
        classes_per_client=3
    )
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    results = {
        'IID': history_iid['test_accuracy'][-1],
        'Non-IID': history_non_iid['test_accuracy'][-1],
        'Non-IID + Common': history_non_iid_common['test_accuracy'][-1]
    }
    
    print("\nFinal Test Accuracies:")
    for name, acc in results.items():
        print(f"  {name:<20}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history_iid['rounds'], history_iid['test_accuracy'], 
           label='IID', marker='o', linewidth=2)
    ax.plot(history_non_iid['rounds'], history_non_iid['test_accuracy'], 
           label='Non-IID', marker='s', linewidth=2)
    ax.plot(history_non_iid_common['rounds'], history_non_iid_common['test_accuracy'], 
           label='Non-IID + Common', marker='^', linewidth=2)
    
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Federated Learning: Comparison of Approaches', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Comparison plot saved as 'final_comparison.png'")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("\n1. IID vs Non-IID:")
    improvement = (results['IID'] - results['Non-IID']) * 100
    print(f"   IID performs {improvement:+.2f}% better than Non-IID")
    print("   This demonstrates the challenge of heterogeneous data in federated learning.")
    
    print("\n2. Impact of Common Dataset:")
    improvement = (results['Non-IID + Common'] - results['Non-IID']) * 100
    print(f"   Common dataset improves Non-IID by {improvement:+.2f}%")
    print("   Fine-tuning on a small uniform dataset helps mitigate Non-IID challenges.")
    
    print("\n3. Practical Recommendations:")
    print("   • Use IID data when possible (though unrealistic in many applications)")
    print("   • For Non-IID scenarios, consider:")
    print("     - Common dataset fine-tuning")
    print("     - More local epochs per round")
    print("     - Advanced algorithms like FedProx")
    print("     - Personalization techniques")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • Data distribution plots")
    print("  • Training curve plots")
    print("  • Saved model files")
    print("  • Final comparison plot")
    print()


if __name__ == "__main__":
    main()

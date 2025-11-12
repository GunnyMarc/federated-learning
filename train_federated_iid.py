"""
Example: Federated Learning with IID Data Distribution

This script demonstrates federated learning with uniformly distributed (IID) data
across clients using the CIFAR-10 dataset.
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Import our federated learning modules
import sys
sys.path.append('..')

from models import SimpleCNN, get_model
from algorithms import FedAvg, get_algorithm
from data_utils import create_federated_dataloaders
from server import FederatedServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning with IID Data')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store data')
    
    # Federated learning arguments
    parser.add_argument('--num_clients', type=int, default=8,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=20,
                       help='Number of communication rounds')
    parser.add_argument('--clients_per_round', type=int, default=None,
                       help='Clients per round (None = all clients)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='Number of local epochs')
    parser.add_argument('--algorithm', type=str, default='fedavg',
                       choices=['fedsgd', 'fedavg', 'fedprox'],
                       help='Federated learning algorithm')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_nn', 'simple_cnn', 'improved_cnn'],
                       help='Model architecture')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save the trained model')
    parser.add_argument('--plot_results', action='store_true',
                       help='Plot training results')
    
    return parser.parse_args()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    axes[0].plot(history['rounds'], history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['rounds'], history['test_loss'], label='Test Loss', marker='s')
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['rounds'], history['test_accuracy'], label='Test Accuracy', 
                marker='o', color='green')
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("Federated Learning with IID Data Distribution")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Number of clients: {args.num_clients}")
    print(f"  Number of rounds: {args.num_rounds}")
    print(f"  Clients per round: {args.clients_per_round or 'All'}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    print()
    
    # Create federated data loaders (IID distribution)
    print("Creating federated data loaders (IID)...")
    train_loaders, test_loader = create_federated_dataloaders(
        num_clients=args.num_clients,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        iid=True,  # IID distribution
        data_dir=args.data_dir
    )
    
    print(f"  Total clients: {len(train_loaders)}")
    print(f"  Samples per client: {[len(loader.dataset) for loader in train_loaders[:3]]}...")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print()
    
    # Determine number of classes
    if args.dataset in ['CIFAR10', 'MNIST', 'FashionMNIST']:
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    else:
        num_classes = 10
    
    # Create model
    print("Initializing model...")
    model = get_model(args.model, num_classes=num_classes)
    print(f"  Model: {args.model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create algorithm
    algorithm = get_algorithm(
        args.algorithm,
        local_epochs=args.local_epochs
    )
    
    # Create server
    print("Initializing federated server...")
    server = FederatedServer(
        model=model,
        algorithm=algorithm,
        num_clients=args.num_clients,
        device=args.device
    )
    print()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("Starting federated training...")
    print("-" * 80)
    
    history = server.train(
        train_loaders=train_loaders,
        test_loader=test_loader,
        criterion=criterion,
        num_rounds=args.num_rounds,
        learning_rate=args.learning_rate,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        eval_every=1,
        verbose=True
    )
    
    print("-" * 80)
    print("\nTraining completed!")
    print()
    
    # Print final results
    final_accuracy = history['test_accuracy'][-1]
    final_loss = history['test_loss'][-1]
    print(f"Final Results:")
    print(f"  Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  Test Loss: {final_loss:.4f}")
    print()
    
    # Save model if requested
    if args.save_model:
        server.save_model(args.save_model)
        print(f"Model saved to {args.save_model}")
        print()
    
    # Plot results if requested
    if args.plot_results:
        plot_save_path = f"federated_iid_{args.algorithm}_results.png"
        plot_training_history(history, save_path=plot_save_path)
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Quick Start Example for Federated Learning

This is the simplest way to get started with federated learning.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_nn import SimpleNN
from src.data.data_distribution import DataDistributor
from src.algorithms.fedavg import FedAvgTrainer


def quick_start():
    """
    Quick start example for federated learning.

    This demonstrates the minimal code needed to:
    1. Load and distribute data
    2. Create a federated learning system
    3. Train the model
    4. Evaluate results
    """

    print("=" * 60)
    print("FEDERATED LEARNING QUICK START")
    print("=" * 60)

    # Configuration
    NUM_CLIENTS = 8
    NUM_ROUNDS = 10
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  - Number of clients: {NUM_CLIENTS}")
    print(f"  - Training rounds: {NUM_ROUNDS}")
    print(f"  - Device: {DEVICE}")

    # Step 1: Prepare data
    print("\n[Step 1] Preparing data...")
    distributor = DataDistributor('cifar10', data_dir='./data')

    client_loaders, test_loader = distributor.create_iid_distribution(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE
    )

    print(f"  ✓ Created {NUM_CLIENTS} clients")
    print(f"  ✓ Each client has ~{len(client_loaders[0].dataset)} samples")
    print(f"  ✓ Test set has {len(test_loader.dataset)} samples")

    # Step 2: Create model
    print("\n[Step 2] Creating model...")
    model = SimpleNN(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {num_params:,} parameters")

    # Step 3: Initialize federated learning trainer
    print("\n[Step 3] Initializing federated learning system...")
    trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        learning_rate=0.01,
        device=DEVICE
    )
    print("  ✓ FedAvg trainer initialized")

    # Step 4: Train
    print("\n[Step 4] Training federated model...")
    print("-" * 60)

    history = trainer.train(
        num_rounds=NUM_ROUNDS,
        local_epochs=1,
        client_fraction=1.0,
        verbose=True
    )

    # Step 5: Results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    final_accuracy = history['test_accuracy'][-1]
    final_loss = history['test_loss'][-1]

    print(f"\nFinal Results:")
    print(f"  - Test Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"  - Test Loss: {final_loss:.4f}")

    print(f"\nProgress:")
    print(f"  - Initial Accuracy: {history['test_accuracy'][0]:.4f}")
    print(f"  - Final Accuracy: {history['test_accuracy'][-1]:.4f}")
    print(f"  - Improvement: {(history['test_accuracy'][-1] - history['test_accuracy'][0]) * 100:.2f}%")

    # Step 6: Save model (optional)
    print("\n[Step 6] Saving model...")
    trained_model = trainer.get_global_model()
    torch.save(trained_model.state_dict(), 'federated_model.pth')
    print("  ✓ Model saved as 'federated_model.pth'")

    print("\n" + "=" * 60)
    print("SUCCESS! Your first federated learning model is trained!")
    print("=" * 60)

    return trainer, history


def load_and_test():
    """
    Example of loading a trained model and testing it.
    """
    print("\nLoading and testing saved model...")

    # Load model
    model = SimpleNN(num_classes=10)
    model.load_state_dict(torch.load('federated_model.pth'))
    model.eval()

    print("✓ Model loaded successfully!")

    # Test on a few samples
    print("\nModel is ready for inference!")

    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run quick start
    trainer, history = quick_start()

    # Optionally load and test
    # model = load_and_test()

    print("\nNext steps:")
    print("  1. Try modifying NUM_CLIENTS or NUM_ROUNDS")
    print("  2. Experiment with Non-IID data distribution")
    print("  3. Try the FedSGD algorithm instead of FedAvg")
    print("  4. Compare with traditional centralized training")
    print("\nSee examples/ directory for more advanced usage!")
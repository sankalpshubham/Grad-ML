import os
import re
import json
import matplotlib.pyplot as plt


def parse_test_out(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    models = []
    model_blocks = content.strip().split("_" * 50)  # Split models using the delimiter

    for block in model_blocks:
        if not block.strip():
            continue

        model_info = {}

        # Extract hyperparameters
        model_info["hidden_dim"] = int(re.search(r"Hidden Dimension: (\d+)", block).group(1))
        model_info["epochs"] = int(re.search(r"Number of Epochs: (\d+)", block).group(1))
        model_info["learning_rate"] = re.search(r"Learning Rate: (\w+)", block).group(1)  # static assumed
        model_info["optimizer"] = re.search(r"Optimizer: (\w+)", block).group(1)

        # Extract final training and validation accuracies
        model_info["final_train_acc"] = float(re.search(r"Final Training Accuracy: ([0-9.]+)", block).group(1))
        model_info["final_val_acc"] = float(re.search(r"Final Validation Accuracy: ([0-9.]+)", block).group(1))

        # Extract total training time
        model_info["total_training_time"] = float(re.search(r"Total Training Time: ([0-9.]+)", block).group(1))

        # Extract per epoch data
        per_epoch_data = re.findall(r"Epoch (\d+): Train Acc = ([0-9.]+), Val Acc = ([0-9.]+), Train Loss = ([0-9.]+), Val Loss = ([0-9.]+)", block)
        model_info["per_epoch_data"] = []
        for epoch, train_acc, val_acc, train_loss, val_loss in per_epoch_data:
            model_info["per_epoch_data"].append({
                "epoch": int(epoch),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss)
            })

        models.append(model_info)

    return models


def compute_average_accuracy_and_loss(models):
    for model in models:
        avg_train_acc = sum(epoch["train_acc"] for epoch in model["per_epoch_data"]) / model["epochs"]
        avg_val_acc = sum(epoch["val_acc"] for epoch in model["per_epoch_data"]) / model["epochs"]
        avg_train_loss = sum(epoch["train_loss"] for epoch in model["per_epoch_data"]) / model["epochs"]
        avg_val_loss = sum(epoch["val_loss"] for epoch in model["per_epoch_data"]) / model["epochs"]

        model["avg_train_acc"] = avg_train_acc
        model["avg_val_acc"] = avg_val_acc
        model["avg_train_loss"] = avg_train_loss
        model["avg_val_loss"] = avg_val_loss


def find_best_model_by_val_acc(models):
    best_model = max(models, key=lambda model: model["final_val_acc"])
    return best_model


def visualize_performance(models):
    for model in models:
        epochs = [epoch["epoch"] for epoch in model["per_epoch_data"]]
        train_acc = [epoch["train_acc"] for epoch in model["per_epoch_data"]]
        val_acc = [epoch["val_acc"] for epoch in model["per_epoch_data"]]
        train_loss = [epoch["train_loss"] for epoch in model["per_epoch_data"]]
        val_loss = [epoch["val_loss"] for epoch in model["per_epoch_data"]]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, label="Train Accuracy", color="blue", marker='o')
        plt.plot(epochs, val_acc, label="Validation Accuracy", color="green", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Model Performance (Hidden Dim: {model['hidden_dim']}, LR: {model['learning_rate']})")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", color="red", marker='o')
        plt.plot(epochs, val_loss, label="Validation Loss", color="orange", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Over Epochs (Hidden Dim: {model['hidden_dim']}, LR: {model['learning_rate']})")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Parse test.out file
    models = parse_test_out("results/test.out")

    # Compute average accuracy and loss per epoch for each model
    compute_average_accuracy_and_loss(models)

    # Find and print best model by validation accuracy
    best_model = find_best_model_by_val_acc(models)
    print("Best Model by Validation Accuracy:")
    print(f"Hidden Dim: {best_model['hidden_dim']}")
    print(f"Final Validation Accuracy: {best_model['final_val_acc']:.4f}")
    print(f"Final Training Accuracy: {best_model['final_train_acc']:.4f}")
    print(f"Total Training Time: {best_model['total_training_time']:.2f} seconds")

    # Visualize performance for each model
    visualize_performance(models)

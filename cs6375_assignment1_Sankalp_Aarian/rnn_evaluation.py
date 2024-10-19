import matplotlib.pyplot as plt
import numpy as np

# Function to parse the file and separate model outputs
def parse_rnn_results(file_path):
    all_models = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_model = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
    for line in lines:
        if line.startswith('_'):  # Separator for models
            if current_model['epochs']:  # If there is data for the current model
                all_models.append(current_model)
                current_model = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
        elif line[0].isdigit():  # If the line starts with a digit, it's a result line
            parts = line.split()
            epoch = int(parts[0])
            train_loss = float(parts[1])
            train_acc = float(parts[2])
            val_acc = float(parts[3])
            current_model['epochs'].append(epoch)
            current_model['train_loss'].append(train_loss)
            current_model['train_acc'].append(train_acc)
            current_model['val_acc'].append(val_acc)
    
    # Append the last model if exists
    if current_model['epochs']:
        all_models.append(current_model)

    return all_models

# Function to plot the results for each model
def plot_rnn_results(models):
    for idx, model in enumerate(models):
        epochs = model['epochs']
        train_loss = model['train_loss']
        train_acc = model['train_acc']
        val_acc = model['val_acc']

        plt.figure(figsize=(12, 8))

        # Plot train loss
        plt.subplot(3, 1, 1)
        plt.plot(epochs, train_loss, label=f'Model {idx+1} Train Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.legend()

        # Plot train accuracy
        plt.subplot(3, 1, 2)
        plt.plot(epochs, train_acc, label=f'Model {idx+1} Train Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Train Accuracy')
        plt.legend()

        # Plot validation accuracy
        plt.subplot(3, 1, 3)
        plt.plot(epochs, val_acc, label=f'Model {idx+1} Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Main function to load data and plot results
if __name__ == "__main__":
    file_path = 'results/rnn_test.out'
    models = parse_rnn_results(file_path)
    plot_rnn_results(models)

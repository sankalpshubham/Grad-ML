import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.activation(self.W1(input_vector))

        # [to fill] obtain output layer representation
        output_layer = self.W2(hidden_layer)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    # GPU TRAINING ----------------------------------------------------------
    # Detect if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNN(input_dim = len(vocab), h = args.hidden_dim).to(device)
    # ------------------------------------------------------------------------

    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

    # MY ADDED ------------------------------------------------------------------------
    total_train_start_time = time.time()
    final_train_accuracy = 0
    final_validation_accuracy = 0
    per_epoch_data = []
    # ------------------------------------------------------------------------

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)  # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        # MY ADDED  -------------
        epoch_loss = 0
        # --------------------------
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]

                # GPU TRAINING ----------------------------------------------------------
                input_vector = input_vector.to(device)
                gold_label = torch.tensor([gold_label]).to(device)
                # ------------------------------------------------------------------------

                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)

                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), gold_label)
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            # --------------------------
            epoch_loss += loss.item()
            # --------------------------
            loss.backward()
            optimizer.step()
        
        train_accuracy = correct / total
        # --------------------------
        avg_train_loss = epoch_loss / (N // minibatch_size)
        # --------------------------
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        # MY ADDED ------------------------------------------------------------------------
        # Update final train accuracy after the last epoch
        if epoch == args.epochs - 1:
            final_train_accuracy = train_accuracy
        # ------------------------------------------------------------------------

        # Validation
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        # --------------------------
        epoch_val_loss = 0
        # --------------------------
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]

                # GPU TRAINING ----------------------------------------------------------
                input_vector = input_vector.to(device)
                gold_label = torch.tensor([gold_label]).to(device)
                # ------------------------------------------------------------------------

                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)

                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), gold_label)
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            # --------------------------
            epoch_val_loss += loss.item()
            # --------------------------
            
        validation_accuracy = correct / total
        # --------------------------
        avg_val_loss = epoch_val_loss / (N // minibatch_size)
        # --------------------------
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        # MY ADDED ------------------------------------------------------------------------
        # Update final validation accuracy after the last epoch
        if epoch == args.epochs - 1:
            final_validation_accuracy = validation_accuracy

        # Store per-epoch data ------------------------------------------------------------------------
        per_epoch_data.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'train_loss': avg_train_loss,
            'validation_loss': avg_val_loss
        })
        # ------------------------------------------------------------------------

    total_training_time = time.time() - total_train_start_time
    # ------------------------------------------------------------------------

    # print out final training score, validation score, and total training time
    print("Final Training Accuracy: {:.4f}".format(final_train_accuracy))
    print("Final Validation Accuracy: {:.4f}".format(final_validation_accuracy))
    print("Total Training Time: {:.2f} seconds".format(total_training_time))

    # write out to results/test.out
    # Ensure the directory exists
    os.makedirs('results', exist_ok=True)

    with open('results/test.out', 'a') as f:  # 'a' mode ensures appending to the file or creating it if it doesn't exist
        # Write hyperparameters
        f.write("Model Hyperparameters:\n")
        f.write("Hidden Dimension: {}\n".format(args.hidden_dim))
        f.write("Number of Epochs: {}\n".format(args.epochs))
        
        # Optionally, write other hyperparameters here
        # f.write("Learning Rate: {}\n".format(0.01))  # assuming a static learning rate of 0.01
        f.write("Learning Rate: static")
        f.write("Optimizer: SGD\n")
        
        # Write final results
        f.write("Final Training Accuracy: {:.4f}\n".format(final_train_accuracy))
        f.write("Final Validation Accuracy: {:.4f}\n".format(final_validation_accuracy))
        f.write("Total Training Time: {:.2f} seconds\n".format(total_training_time))

        # Write per-epoch results
        f.write("\nPer Epoch Data:\n")
        for epoch_data in per_epoch_data:
            f.write("Epoch {}: Train Acc = {:.4f}, Val Acc = {:.4f}, Train Loss = {:.4f}, Val Loss = {:.4f}\n".format(
                epoch_data['epoch'], epoch_data['train_accuracy'], epoch_data['validation_accuracy'], epoch_data['train_loss'], epoch_data['validation_loss']
            ))
        
        f.write("_"*50)
        f.write("\n")
        
    
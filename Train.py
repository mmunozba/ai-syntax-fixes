#!/usr/bin/python3
"""
Module for training machine learning models.

Arguments
----------
--source
    Folder path where the JSON files for the dataset to be used for training are.
    May also end in .json if only a single JSON is to be used for training.
--destination
    Path to where the models should be saved to.

Example
----------
$ python .\Train.py --source /data/new_dataset/ --destination models/trained-model

"""

import argparse
import torch
import logging
import json
import time
import math
import pickle
import util.IOProcessor as IOProcessor
from Model import RNN
from torch.utils.data import DataLoader
from util.CodeDataset import CodeDataset
from torch import nn, optim
from util.StringUtils import remove_suffix
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

# Select CPU or GPU for Pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force CPU


def train(model, source, modeltype):
    """
    Trains the given model on the given dataset.
    Note: This process may take multiple hours depending on the model and size of the dataset.

    Parameters
    ----------
    model : RNN
        RNN model to be trained.
    source : str
        Folder path where the JSON files for the dataset to be used for training are.
        May also end in .json if only a single JSON is to be used for training.
    modeltype : int
        Index corresponding to the modeltype.
        Can be 0 (fixlocation), 1 (fixtype), 2 (fixinsert), 3 (fixmodify).

    Returns
    -------
    RNN
        The trained RNN.

    """
    
    data = [[], []]
    raw_training_samples = []

    # Load samples
    if source.endswith(".json"):  # Single JSON file
        with open(source) as file:
            logging.info("Source ending in .json. Predicting on single JSON file.")
            raw_training_samples = json.load(file)
    else:  # Folder path
        for filename in listdir(source):
            with open(source + filename) as file:
                raw_training_samples.extend(json.load(file))
    
    # Manual selection of data range [0-98], one JSON for evaluation
    # for i in range(0,99):
    #     with open(f"../data/new_dataset/training_{i}.json") as file:
    #         raw_training_samples.extend(json.load(file))  

    # Create list with input label pairs
    for sample in raw_training_samples:
        try:                
            actual_sample, tokens = IOProcessor.preprocess(sample["wrong_code"])

            if modeltype == 0 or modeltype == 1:    # Always add all samples
                label = IOProcessor.preprocess_label(sample, modeltype, tokens)
                data[0].append(actual_sample)
                data[1].append(label)
            if modeltype == 2:                      # Only train on insert samples
                if sample["metadata"]["fix_type"] == "insert":
                    label = IOProcessor.preprocess_label(sample, modeltype, tokens)
                    data[0].append(actual_sample)
                    data[1].append(label)
            if modeltype == 3:                      # Only train on modify samples
                if sample["metadata"]["fix_type"] == "modify":
                    label = IOProcessor.preprocess_label(sample, modeltype, tokens)
                    data[0].append(actual_sample)
                    data[1].append(label)
        except Exception as e:
            # Samples where the preprocessing fails are ignored and not used for training
            logging.warning(f"{e} occurred.")
            logging.warning(f"Preprocessing failed for {sample['metadata']['id']}.")
            logging.warning("Skipping the sample.")
    return train_step(model, data)


def train_step(model, data):
    """
    Trains the given model on the given data.
    Adapted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    Parameters
    ----------
    model : RNN
        RNN model to be trained.
    data : list
        List with samples and their corresponding labels.
        Here, data[0] should contain the preprocessed samples and data[1] the corresponding labels.

    Returns
    -------
    model : RNN
        The trained RNN.

    """
    start = time.time()  # Used to calculate training time
    
    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.005)
    epochs = 30
    
    # Preparing the dataset for training
    dataset = CodeDataset(data[0], data[1])
    
    # Splitting dataset from https://stackoverflow.com/a/51768651
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(trainloader, model, criterion, optimizer)
        test_loop(testloader, model, criterion)

    print('Finished Training')
    print(f"Time taken to train: {time_since(start)}")
    return model


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Performs a single training step with the given model on a single batch of data.
    Adapted from from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader containing the training dataset.
    model : RNN
        RNN model to be trained.
    loss_fn
        Criterion to be used to calculate loss. E.g. nn.CrossEntropyLoss()
    optimizer
        Optimizer to be used for optimizing the model parameters. E.g. optim.Adamax()

    Returns
    -------
    None.

    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        
        pred = []
        for sample in X:
            hidden = model.initHidden()
            for word_index in range(sample.size()[0]):
                # # Stop at end of token sequence (start of 0 padding)
                # if sample[word_index].item() == 0:
                #     break
                output, hidden = model(sample[word_index], hidden)
            pred.append(output[0][0])
        
        pred_tensor = torch.stack(pred)
        
        loss = loss_fn(pred_tensor, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    """
    Performs a single test step with the given model on the given dataset.
    Adapted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader containing the test dataset.
    model : RNN
        RNN model to be tested.
    loss_fn
        Criterion to be used to calculate loss. E.g. nn.CrossEntropyLoss()

    Returns
    -------
    None.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            pred = []
            for sample in X:
                hidden = model.initHidden()
                for word_index in range(sample.size()[0]):
                    output, hidden = model(sample[word_index], hidden)
                pred.append(output[0][0])
            
            pred_tensor = torch.stack(pred)
            test_loss += loss_fn(pred_tensor, y).item()
            correct += (pred_tensor.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def save_model(model, destination):
    """
    Saves the given model to the destination.

    Parameters
    ----------
    model : RNN
        RNN model to be saved.
    destination : str
        String of the destination path.

    Returns
    -------
    None.

    """
    logging.info("Saving model to disk.")
    torch.save(model, destination)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    
    with open('util/vocabulary.pickle', 'rb') as file:
        vocabulary = pickle.load(file)
    
    destination = args.destination
    if destination.endswith(".pth"):
        logging.info("Given destination ending in .pth. Removing suffix.")
        destination = remove_suffix(destination, ".pth")
    
    # Train and save four models
    
    # Fix Location Model
    fixlocationmodel = RNN(vocabulary.n_words, 128, 50)
    fixlocationmodel.to(device)
    train(fixlocationmodel, args.source, 0)
    save_model(fixlocationmodel, destination + "-0.pth")   
    
    # Fix Type Model
    fixtypemodel = RNN(vocabulary.n_words, 128, 3)
    fixtypemodel.to(device)
    train(fixtypemodel, args.source, 1)
    save_model(fixtypemodel, destination + "-1.pth")
    
    # Fix Insert Model
    fixinsertmodel = RNN(vocabulary.n_words, 128, 99)
    fixinsertmodel.to(device)
    train(fixinsertmodel, args.source, 2)
    save_model(fixinsertmodel,destination + "-2.pth")
    
    # Fix Modify Model
    fixmodifymodel = RNN(vocabulary.n_words, 128, 99)
    fixmodifymodel.to(device)
    train(fixmodifymodel, args.source, 3)
    save_model(fixmodifymodel, destination + "-3.pth")

#!/usr/bin/python3
"""
Module for running predictions with machine learning models.

Arguments
----------
--model
    Path to the trained models. Note: This is not the full path (ending in .pth) 
    but the folder path + model name. For example "folder/modelname" loads the 
    models "folder/modelname-0.pth", "folder/modelname-1.pth" etc.
--source
    Folder path where the input JSON files are located. 
    May also end in .json when only a single JSON is to be used for predictions.
--destination
    File path where the prediction output will be saved. Should end in ".json".

Example
----------
$ python .\Predict.py --model models/final-model --source input.json --destination output.json

"""

import argparse
import torch
import logging
import json
import util.IOProcessor as IOProcessor
from util.StringUtils import remove_suffix
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Folder path of all test files.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted predictions.", required=True)

# Select CPU or GPU for Pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force CPU


def predict(models, test_files):
    """
    Returns a set of predictions on the given set of files using the given models.

    Parameters
    ----------
    models : list of RNN
        List of four RNN models for a full prediction.
    test_files : str
        Path to folder containing JSON files with the samples.

    Returns
    -------
    predictions : list of Dict
        List of predictions. Each prediction consists of metadata and 
        the predicted fix_location, fix_type and fix_token (if applicable).

    """
    predictions = []
    raw_training_samples = []

    # Load samples
    if test_files.endswith(".json"):  # Single JSON file
        with open(test_files) as file:
            logging.info("Source ending in .json. Predicting on single JSON file.")
            raw_training_samples = json.load(file)
    else:  # Folder path
        for filename in listdir(test_files):
            with open(test_files + filename) as file:
                raw_training_samples.extend(json.load(file))

    # Retrieve models from input and move to device
    fixlocationmodel = models[0]
    fixtypemodel = models[1]
    fixinsertmodel = models[2]
    fixmodifymodel = models[3]
    fixlocationmodel.to(device)
    fixtypemodel.to(device)
    fixinsertmodel.to(device)
    fixmodifymodel.to(device)

    for sample in raw_training_samples:
        try:
            logging.info(f"Running prediction on {sample['metadata']['file']}.")
            actual_sample, tokens = IOProcessor.preprocess(sample["wrong_code"])
    
            with torch.no_grad():
                                
                # Predict Fix Type and Location
                predicted_token_location = predict_single(actual_sample, fixlocationmodel)
                predicted_location = IOProcessor.postprocess(predicted_token_location, 0, tokens, sample["wrong_code"])
                try:
                    predicted_token_old = tokens[predicted_token_location]
                except IndexError:  # If predicted index is outside of token range
                    predicted_token_old = ""
                predicted_type = IOProcessor.postprocess(
                    predict_single(actual_sample, fixtypemodel), 1)
                predicted_token = ""
    
                # Predict Fix Token, if needed
                if predicted_type == "insert":
                    predicted_token = IOProcessor.postprocess(
                        predict_single(actual_sample, fixinsertmodel), 2)
                if predicted_type == "modify":
                    predicted_token = IOProcessor.postprocess(
                        predict_single(actual_sample, fixmodifymodel), 3)
                    
            # Build the list of predictions
            prediction = IOProcessor.buildPredictionJson(sample, predicted_location, predicted_type, predicted_token_location, predicted_token_old, predicted_token)
            predictions.append(prediction)

        except Exception as e:
            logging.warning(f"{e.__class__.__name__} occurred: {e}")
            logging.warning(f"Prediction failed for {sample['metadata']['id']}.")
            logging.warning("Skipping the sample.")
            prediction = {}

    return predictions


def load_model(source):
    """
    Loads the four models necessary for a prediction located at the specified source path.
    Note: This is not the full path (ending in .pth) but the folder path + model name.
    Usage: load_model("folder/modelname") loads the models "folder/modelname-0.pth", "folder/modelname-1.pth" etc.

    Parameters
    ----------
    source : str
        File path leading to the set of models.
        Each model should be at source + "-X.pth". With X being the modeltype index.

    Returns
    -------
    models : list of RNN
        List of four models loaded from disk.

    """

    models = []

    for index in range(4):
        logging.info("Loading model from disk.")
        if source.endswith(".pth"):
            logging.warning("Given file path ended with .pth. Removing suffix and attempting to load.")
            source = remove_suffix(source, ".pth")
        model = torch.load(source + f"-{index}.pth", map_location=device)
        model.to(device)
        model.eval()
        models.append(model)

    return models


def write_predictions(destination, predictions):
    """
    Takes a list of predictions and writes it to a JSON file at the destination.

    Parameters
    ----------
    destination : str
        Destination file path for the JSON output.
    predictions : list of Dict
        List of predictions to be written.

    Returns
    -------
    None.

    """

    logging.info("Writing predictions to disk.")

    cleaned_predictions = []

    for index, prediction in enumerate(predictions):
        # Remove correct label info from predictions
        cleaned_prediction = {
            "metadata": {
                "file": prediction["metadata"]["file"],
                "id": prediction["metadata"]["id"],
                # "wrong_code": prediction["correct_data"]["wrong_code"],
                # "correct_code": prediction["correct_data"]["correct_code"],
                # "fix_location": prediction["correct_data"]["correct_location"],
                # "fix_type": prediction["correct_data"]["correct_type"],
            },
            "predicted_location": prediction["predicted_location"],
            "predicted_type": prediction["predicted_type"],
        }

        if prediction["predicted_type"] == "modify" or prediction["predicted_type"] == "insert":
            cleaned_prediction["predicted_token"] = prediction["predicted_token"]

        cleaned_prediction["predicted_code"] = prediction["predicted_code"]

        cleaned_predictions.append(cleaned_prediction)

    # Write summary JSON
    with open(destination, 'w') as file:
        json.dump(cleaned_predictions, file, indent=2)


def predict_single(actual_sample, model):
    """
    Takes a sample and returns a single prediction with the given model.

    Parameters
    ----------
    actual_sample : list of int
        List of tokens to be used for a prediction.
    model : RNN
        RNN model to be used for the prediction.

    Returns
    -------
    label_id : int
        Label predicted by the ML model.
        Depending on the modeltype, this can be
        - a token index ([0-n_tokens] fixlocation)
        - a fix type ([0-2] fixtype)
        - a fix token ([0-99] fixtoken)

    """
    hidden = model.initHidden()
    for word_index in range(actual_sample.size()[0]):
        output, hidden = model(actual_sample[word_index], hidden)
    _, predicted = torch.max(output[0][0], 0)
    label_id = predicted.item()
    return label_id


if __name__ == "__main__":
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)  # Uncomment for info logs

    # load the serialized model
    model = load_model(args.model)

    # predict incorrect location for each test example.
    predictions = predict(model, args.source)

    # write predictions to file
    write_predictions(args.destination, predictions)

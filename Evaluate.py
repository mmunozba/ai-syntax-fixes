#!/usr/bin/python3
"""
Module for evaluating machine learning models.

Arguments
----------
--model
    Path to the trained models. Note: This is not the full path (ending in .pth)
    but the folder path + model name. For example "folder/modelname" loads the
    models "folder/modelname-0.pth", "folder/modelname-1.pth" etc.
--source
    Folder path where the input JSON files are located.
    May also end in .json when only a single JSON is to be used for evaluation.

Example
----------
$ python .\Evaluate.py --model models/final-model --source input.json

"""

import logging
import argparse
import libcst as cst
from Predict import load_model, predict, predict_single
from util.FixType import FixType
import util.IOProcessor as IOProcessor
from os import listdir
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Folder path of all test files.", required=True)


def evaluate(model, test_files):
    """
    Evaluates the given models on the given set of samples.
    Results are printed on the console.

    Parameters
    ----------
    model : str
        Path to the set of models to be used for predictions.
    test_files : str
        File path to the JSON with the set of samples.
        May also be a folder path to a folder containing multiple JSONs.

    Returns
    -------
    None.

    """
    print("Running predictions.")
    models = load_model(model)
    predictions = predict(models, test_files)

    # # write predictions to file
    # write_predictions("evaluate_out.json",predictions)
    evaluate_individual(predictions, test_files, models)
    evaluate_overall(predictions)


def evaluate_individual(predictions, test_files, models):
    """
    Evaluates each of the given models individually.
    Results are printed on the console.

    Parameters
    ----------
    predictions : list of Dict
        List of outputs of the predict() function from the Predict module.
    test_files : str
        File path to the JSON with the set of samples.
        May also be a folder path to a folder containing multiple JSONs.
    models : list of RNN
        List of four RNN models to be used for predictions.

    Returns
    -------
    None.

    """

    print("\nAccuracy for individual models\n")
    
    # Fix Location
    correct_predictions = [0, 0, 0]
    total_predictions = [0, 0, 0]
    num_failed_predictions = 0

    for prediction in predictions:
        if prediction["correct_data"]["correct_location"] == prediction["predicted_location"]:
            correct_predictions[FixType[prediction["correct_data"]["correct_type"]].value] = correct_predictions[FixType[
                prediction["correct_data"]["correct_type"]].value] + 1
        if prediction["predicted_location"] is None:
            num_failed_predictions = num_failed_predictions + 1
        total_predictions[FixType[prediction["correct_data"]["correct_type"]].value] = total_predictions[FixType[
            prediction["correct_data"]["correct_type"]].value] + 1

    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = correct_predictions[i] / total_predictions[i]
        print(f"Fix Location accuracy for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(correct_predictions) / (len(predictions) - num_failed_predictions)
    print(f"Fix Location accuracy overall is {accuracy * 100} %")
    
    # Fix type
    correct_predictions = [0, 0, 0]
    total_predictions = [0, 0, 0]
    num_failed_predictions = 0

    for prediction in predictions:
        if prediction["correct_data"]["correct_type"] == prediction["predicted_type"]:
            correct_predictions[FixType[prediction["predicted_type"]].value] = correct_predictions[FixType[
                prediction["predicted_type"]].value] + 1
        if prediction["predicted_type"] is None:
            num_failed_predictions = num_failed_predictions + 1
        total_predictions[FixType[prediction["predicted_type"]].value] = total_predictions[FixType[
            prediction["predicted_type"]].value] + 1

    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = correct_predictions[i] / total_predictions[i]
        print(f"Fix Type accuracy for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(correct_predictions) / (len(predictions) - num_failed_predictions)
    print(f"Fix Type accuracy overall is {accuracy * 100} %")
    
    # We repeat the predictions to evaluate the insert and modify models individually, regardless of the predicted fix type 

    raw_training_samples = []

    if test_files.endswith(".json"):  # Single JSON file
        with open(test_files) as file:
            logging.info("Source ending in .json. Predicting on single JSON file.")
            raw_training_samples = json.load(file)
    else:  # Folder path
        for filename in listdir(test_files):
            with open(test_files + filename) as file:
                raw_training_samples.extend(json.load(file))
    
    correct_predictions_insert = 0
    total_predictions_insert = 0
    correct_predictions_modify = 0
    total_predictions_modify = 0
    insert_tokens = []
    modify_tokens = []

    for sample in raw_training_samples:
        # Insert
        if sample["metadata"]["fix_type"] == "insert":
            actual_sample, tokens = IOProcessor.preprocess(sample["wrong_code"])
            pred = predict_single(actual_sample, models[2])
            token = IOProcessor.postprocess(pred, 2)
            if token == sample["metadata"]["fix_token"]: # Correct Prediction
                correct_predictions_insert = correct_predictions_insert + 1
            else:                                  # Incorrect prediction
                insert_tokens.append([token, sample["metadata"]["fix_token"]])
            total_predictions_insert = total_predictions_insert + 1
        # Modify
        if sample["metadata"]["fix_type"] == "modify":
            actual_sample, tokens = IOProcessor.preprocess(sample["wrong_code"])
            pred = predict_single(actual_sample, models[3])
            token = IOProcessor.postprocess(pred, 3)
            if token == sample["metadata"]["fix_token"]: # Correct Prediction
                correct_predictions_modify = correct_predictions_modify + 1
            else:                                  # Incorrect prediction
                modify_tokens.append([token, sample["metadata"]["fix_token"]])
            total_predictions_modify = total_predictions_modify + 1

    insert_accuracy = correct_predictions_insert / total_predictions_insert
    modify_accuracy = correct_predictions_modify / total_predictions_modify
    print(f"Fix Token accuracy for insert is {insert_accuracy * 100} %")
    print(f"Fix Token accuracy for modify is {modify_accuracy * 100} %")

    # The following code may be used to create a swarm plot of the erroneous predictions for fix locations
    # This does, however, require the installation of the pandas, seaborn, and matplotlib libraries.
    
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # location_distance_array = []
    # for prediction in predictions:
    #     actual_sample, tokens = IOProcessor.preprocess(prediction["correct_data"]["wrong_code"])
    #     label = get_token_index(prediction["correct_data"]["wrong_code"], tokens, prediction["correct_data"]["correct_location"])
    #     if prediction["predicted_token_location"] - label == 0:
    #         pass
    #     else:
    #         location_distance_array.append([prediction["predicted_token_location"] - label, prediction["correct_data"]["correct_type"]])
    
    # df = pd.DataFrame(data=location_distance_array)
    # sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 4))
    # sns.despine(bottom=True, left=True)
    # sns.swarmplot(y=0, x=1, data=df, palette="dark", size=6)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # plt.ylim([-15, 16])
    
    # plt.savefig('line_plot.pdf', bbox_inches='tight', pad_inches=0)


def evaluate_overall(predictions):
    """
    Evaluates the given predictions based on the overall task.
    Results are printed on the console.

    Parameters
    ----------
    predictions : list of Dict
        List of outputs of the predict() function from the Predict module.

    Returns
    -------
    None.

    """
    print("\nAccuracy for the the models combined (full prediction):\n")

    perfect_predictions = [0, 0, 0]
    exact_code_match = [0, 0, 0]
    exact_code_match_ignore_spaces = [0, 0, 0]
    correct_syntax = [0, 0, 0]
    total_predictions = [0, 0, 0]
    num_failed_predictions = 0
    
    for prediction in predictions:
        # Perfect Prediction
        if prediction["correct_data"]["correct_location"] == prediction["predicted_location"]:
            # Correct type
            if prediction["correct_data"]["correct_type"] == prediction["predicted_type"]:
                # Correct token
                if prediction["predicted_type"] == "insert" or prediction["predicted_type"] == "modify":
                    if "correct_token" in prediction["correct_data"] and prediction["correct_data"]["correct_token"] == \
                            prediction["predicted_token"]:
                        perfect_predictions[FixType[prediction["predicted_type"]].value] = perfect_predictions[FixType[
                            prediction["predicted_type"]].value] + 1
                else:
                    perfect_predictions[FixType[prediction["predicted_type"]].value] = perfect_predictions[FixType[
                        prediction["predicted_type"]].value] + 1
        
        # Exact Code Match
        if prediction["predicted_code"] == prediction["correct_data"]["correct_code"]:
            exact_code_match[FixType[prediction["predicted_type"]].value] = exact_code_match[FixType[
                prediction["predicted_type"]].value] + 1
            
        # Exact Code Match (Ignore Spaces)
        if prediction["predicted_code"].replace(" ", "") == prediction["correct_data"]["correct_code"].replace(" ", ""):
            exact_code_match_ignore_spaces[FixType[prediction["predicted_type"]].value] = exact_code_match_ignore_spaces[FixType[
                prediction["predicted_type"]].value] + 1
        
        # Correct Syntax Fix
        try:
            cst.parse_module(prediction["predicted_code"])
            correct_syntax[FixType[prediction["predicted_type"]].value] = correct_syntax[FixType[
                prediction["predicted_type"]].value] + 1
        except Exception as e:
            logging.warning(f"{e.__class__.__name__} occurred: {e}")
            # Happens if parsing fails
            pass
        
        total_predictions[FixType[prediction["predicted_type"]].value] = total_predictions[FixType[
            prediction["predicted_type"]].value] + 1
    
    # Perfect Prediction
    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = perfect_predictions[i] / total_predictions[i]
        print(f"Perfect Prediction for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(perfect_predictions) / (len(predictions) - num_failed_predictions)
    print(f"Perfect Prediction accuracy overall is {accuracy * 100} %")
    
    # Exact Code Match
    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = exact_code_match[i] / total_predictions[i]
        print(f"Exact Code Match for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(exact_code_match) / (len(predictions) - num_failed_predictions)
    print(f"Exact Code Match accuracy overall is {accuracy * 100} %")
    
    # Exact Code Match (Ignore Spaces)
    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = exact_code_match_ignore_spaces[i] / total_predictions[i]
        print(f"Exact Code Match (Ignore Spaces) for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(exact_code_match_ignore_spaces) / (len(predictions) - num_failed_predictions)
    print(f"Exact Code Match (Ignore Spaces) accuracy overall is {accuracy * 100} %")
    
    # Correct Syntax Fixes
    for i in range(3):
        if total_predictions[i] == 0:  # If the type was never predicted
            accuracy = 0
        else:
            accuracy = correct_syntax[i] / total_predictions[i]
        print(f"Correct Syntax Fixes for class {FixType(i).name}: {accuracy * 100} %")

    accuracy = sum(correct_syntax) / (len(predictions) - num_failed_predictions)
    print(f"Correct Syntax Fixes accuracy overall is {accuracy * 100} %")


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)  # Set log level
    evaluate(args.model, args.source)

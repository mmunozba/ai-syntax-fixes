# Fixing Syntax Errors Using Deep Learning

A CLI application written in Python that locates and fixes syntax errors in source code using the PyTorch library and deep learning with reccurent neural networks.
This project was created as part of the "Analyzing Software with Deep Learning" course at the University of Stuttgart.

For detailed information on the architecture and model performance please check the [project paper](project-report.pdf).


## Contents

```
.
├── models/
│   ├── final-model-0.pth
│   ├── final-model-1.pth
│   ├── final-model-2.pth
│   └── final-model-3.pth
├── util/
│   └── ...
├── Evaluate.py
├── Model.py
├── Predict.py
├── Train.py
└── ...
```

- `models/`: Contains a single set of four trained models.
- `util/`: Various utility scripts used by the main programs.
- `Evaluate.py`: Used to evaluate a set of models.
- `Model.py`: PyTorch model architecture used by all four models.
- `Predict.py`: Used for predictions on a set of samples.
- `Train.py`: Used for training a new set of four models on a set of samples.

## Prerequisites

If you already have PyTorch and LibCST installed you can skip the second step.

1. Install Python 3.8 and PIP.
2. Run `pip install -r requirements.txt`. This installs the following:
    - Pytorch 1.8.1 needed for machine learning.
    - LibCST used by `Evaluate.py` to check code syntax fixes.

## Usage

The commands given in this section were tested on Windows. For Linux systems, the commands for running a Python script might be slightly different.

### Prediction
To use a trained set of models for a prediction, run the following:

`python Predict.py --model MODEL --source SOURCE --destination DESTINATION.json`

Arguments:
- `MODEL`: Path to the trained models. Note: This is not the full path (ending in .pth) but the folder path + model name. For example "folder/modelname" loads the models "folder/modelname-0.pth", "folder/modelname-1.pth" etc.
- `SOURCE`: Folder path where the input JSON files are located. May also end in .json when only a single JSON is to be used for predictions.
- `DESTINATION.json`: File path where the prediction output will be saved. Should end in ".json".

Examples:
```bash
python Predict.py --model models/final-model --source data/ --destination output.json
python Predict.py --model models/final-model --source data/input.json --destination output.json
```

### Training

Warning: Running the training function trains four complete new RNN models.
Depending on the size of the training dataset, training these models may take anywhere from a few minutes to multiple hours.

To train a new set of models on a dataset, run the following:

`python Train.py --source SOURCE --destination DESTINATION`

Arguments:
- `SOURCE`: Folder path where the JSON files for the dataset to be used for training are. May also end in .json if only a single JSON is to be used for training.
- `DESTINATION`: Path to where the models should be saved to.

Examples:
```bash
python Train.py --source data/new_dataset/ --destination models/trained-model
python Train.py --source data/input.json --destination models/trained-model
```

### Evaluation
To evaluate a set of trained models on a dataset, run the following:

`python Evaluate.py --model MODEL --source SOURCE`

Arguments:
- `MODEL`: Path to the trained models. Note: This is not the full path (ending in .pth) but the folder path + model name. For example "folder/modelname" loads the models "folder/modelname-0.pth", "folder/modelname-1.pth" etc.
- `SOURCE`: Folder path where the input JSON files are located. May also end in .json when only a single JSON is to be used for evaluation.

Examples:
```bash
python Evaluate.py --model models/final-model --source data/
python Evaluate.py --model models/final-model --source data/input.json
```

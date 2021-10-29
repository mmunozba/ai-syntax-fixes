"""
Contains helper functions for pre- and postprocessing data for the machine learning models.
"""
import torch
import pickle
from util.FixType import FixType
from util.Tokenizer import to_token_list_from_string, get_character_index, get_token_index
from util.StringUtils import replace_nth

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load token vocabulary
with open('util/vocabulary.pickle', 'rb') as file:
    vocabulary = pickle.load(file)    


def preprocess(codestring):
    """
    Takes a string of Python code and returns a tensor ready to be used by 
    the ML model and a list of human-readable tokens.

    Parameters
    ----------
    codestring : str
        Python code as a string.

    Returns
    -------
    actual_sample : Tensor
        Tensor with tokens for the ML model, padded to 50 tokens.
    tokens : list of str
        List of human-readable tokens.

    """
    tokens = to_token_list_from_string(codestring)
    token_tensor = tensorFromSentence(vocabulary, tokens)
    input_sample = torch.tensor(token_tensor.numpy()).to(device)
    actual_sample = torch.zeros(50, dtype=torch.long).to(device)
    actual_sample[:token_tensor.numpy().size] = input_sample
    return actual_sample, tokens


def preprocess_label(sample, modeltype, tokens):
    """
    Takes a sample and returns the correct label to be used for training with the ML model.

    Parameters
    ----------
    sample : Dict
        Single sample from the JSON dataset.
    modeltype : int
        Index corresponding to the modeltype.
        Can be 0 (fixlocation), 1 (fixtype), 2 (fixinsert), 3 (fixmodify).
    tokens : list of str
        List of human-readable tokens.

    Returns
    -------
    label : int
        Label ready to be used for training with the ML model.
        Depending on the modeltype, this can be
        - a token index ([0-n_tokens] fixlocation)
        - a fix type ([0-2] fixtype)
        - a fix token ([0-98] fixtoken)

    """
    if modeltype == 0:
        label = get_token_index(sample["wrong_code"], tokens, sample["metadata"]["fix_location"])
    if modeltype == 1:
        label = FixType[sample["metadata"]["fix_type"]].value
    if modeltype == 2 or modeltype == 3:
        label = vocabulary.word2index[sample["metadata"]["fix_token"]]
    return label


def postprocess(label_id, modeltype, tokens=[], codestring=""):
    """
    Returns the human-readable label for the given ML prediction output.

    Parameters
    ----------
    label_id : int
        Label predicted by the ML model.
        Depending on the modeltype, this can be
        - a token index ([0-n_tokens] fixlocation)
        - a fix type ([0-2] fixtype)
        - a fix token ([0-98] fixtoken)
    modeltype : int
        Index corresponding to the modeltype.
        Can be 0 (fixlocation), 1 (fixtype), 2 (fixinsert), 3 (fixmodify).
    tokens : list of str, optional
        List of human-readable tokens. The default is [].
    codestring : str, optional
        Python code as a string. The default is "".

    Returns
    -------
    label : str
        Human-readable representation of the predicted label.
        Depending on the modeltype, this can be
        - a character index ([0-c_chars] fixlocation)
        - "delete", "insert", "modify" (fixtype)
        - any token in the vocabulary (fixtoken)

    """
    if modeltype == 0:
        label = get_character_index(codestring, tokens, label_id)
    if modeltype == 1:
        label = FixType(label_id).name
    if modeltype == 2 or modeltype == 3:
        label = vocabulary.index2word[label_id]
    return label


def buildPredictionJson(sample, predicted_location, predicted_type, predicted_token_location, predicted_token_old, predicted_token=""):
    """
    Takes the output of a prediction and returns a dictionary formatted to be saved as an output JSON.

    Parameters
    ----------
    sample : Dict
        Single sample from the JSON dataset.
    predicted_location : str
        Predicted character index of the syntax error. Can be 0-n_chars.
    predicted_type : str
        Predicted fix type. Can be "delete", "insert", "modify".
    predicted_token : str, optional
        Any token in the vocabulary. The default is "".

    Returns
    -------
    prediction : Dict
        Dictionary formatted to be saved as an output JSON.

    """
    # For debugging
    correct_location = sample["metadata"]["fix_location"]
    correct_type = sample["metadata"]["fix_type"]
    correct_code = sample["correct_code"]
    wrong_code = sample["wrong_code"]
    
    prediction = {
        "metadata": {
            "file": sample["metadata"]["file"],
            "id": sample["metadata"]["id"],                
        },
        "predicted_location": predicted_location,
        "predicted_type": predicted_type,
        "predicted_token_location": predicted_token_location,
        "predicted_token_old": predicted_token_old,
        "correct_data":  {
            "correct_location": correct_location,
            "correct_type": correct_type,
            "correct_code": correct_code,
            "wrong_code": wrong_code,

        }
      }
    if predicted_type == "modify" or predicted_type == "insert":
        prediction["predicted_token"] = predicted_token       
    
    if correct_type == "modify" or correct_type == "insert":
        correct_token = sample["metadata"]["fix_token"]
        prediction["correct_data"]["correct_token"] = correct_token
        
    # Write fixed Python code 
    if prediction["predicted_type"] == "insert" or prediction["predicted_type"] == "modify":
        predicted_code = fix_code(prediction["correct_data"]["wrong_code"], prediction["predicted_location"], prediction["predicted_type"], prediction["predicted_token"])
    if prediction["predicted_type"] == "delete":
        predicted_code = fix_code(prediction["correct_data"]["wrong_code"], prediction["predicted_location"], prediction["predicted_type"])
    prediction["predicted_code"] = predicted_code
    
    return prediction


def fix_code(codestring, predicted_location, predicted_type, predicted_token=""):
    """
    Takes a codestring with a syntax error and fixes it with the given prediction.

    Parameters
    ----------
    codestring : str
        Python code as a string.
    predicted_location : str
        Predicted character index of the syntax error. Can be 0-n_chars.
    predicted_type : str
        Predicted fix type. Can be "delete", "insert", "modify".
    predicted_token : str, optional
        Any token in the vocabulary. The default is "".

    Returns
    -------
    fixed_codestring : str
        Fixed version of the code string.

    """
    fixed_codestring = ""
    tokens = to_token_list_from_string(codestring)
    token_index = get_token_index(codestring, tokens,predicted_location)
    
    if predicted_type == "insert":
        # Add spaces if needed
        if predicted_token == "def" or predicted_token == "ID":
            predicted_token = predicted_token + " "
        fixed_codestring = codestring[:predicted_location] + predicted_token + codestring[predicted_location:]
    if predicted_type == "delete":
        if predicted_location == len(codestring):
            fix_code(codestring, predicted_location, "insert", predicted_token)
        else:
            if codestring[(predicted_location + len(tokens[token_index])):].startswith(" "):     # Optimization: Remove space after token
                codestring = codestring[:(predicted_location + len(tokens[token_index]))] + codestring[(predicted_location + len(tokens[token_index]) + 1):]
            token_occurences_before_index = tokens[:token_index].count(tokens[token_index])
            fixed_codestring = replace_nth(codestring, tokens[token_index], predicted_token, token_occurences_before_index + 1)
    if predicted_type == "modify":
        if predicted_location == len(codestring):
            fix_code(codestring, predicted_location, "insert", predicted_token)
        else:
            if predicted_token == "def":                    # Optimization: def tokens always need a space afterwards
                predicted_token = predicted_token + " "
            if predicted_token == "\n":                     # Optimization: Space after \n would cause indendation errors
                if codestring[(predicted_location + len(tokens[token_index])):].startswith(" "):     # Optimization: Remove space after token
                    codestring = codestring[:(predicted_location + len(tokens[token_index]))] + codestring[(predicted_location + len(tokens[token_index]) + 1):]
            token_occurences_before_index = tokens[:token_index].count(tokens[token_index])
            fixed_codestring = replace_nth(codestring, tokens[token_index], predicted_token, token_occurences_before_index + 1)
    
    return fixed_codestring


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence:
        try:
            index = lang.word2index[word]
        except:
            index = 0
        indexes.append(index)
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes)
#!/usr/bin/python3
"""
Contains functionality to tokenize Python code and utilities to handle token lists.
"""

import argparse
import json
import tokenize
from io import StringIO
from util.StringUtils import find_nth
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="File containing Python code to be parsed.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted tokens.", required=True)


def to_token_list(s):
    """
    Read a Python source code file and return the corresponding token list.

    Parameters
    ----------
    s : str
        Path to the Python source code file.

    Returns
    -------
    tokens : list of str
        List of tokens from the source code file.

    """
    tokens = []  # list of tokens extracted from source code.
    
    with tokenize.open(s) as file:
        token_generator = tokenize.generate_tokens(file.readline)
        for token_type, token_string, start_pos, end_pos, line in token_generator:
            tokens.append(token_string)

    # Remove empty strings
    tokens = list(filter(None, tokens))
    return tokens


def to_token_list_from_string(code_string):
    """
    Returns a list of tokens from a Python source code string.

    Parameters
    ----------
    code_string : str
        Python source code string.

    Returns
    -------
    tokens : list of str
        List of tokens that occur in the string.

    """
    tokens = []
    token_generator = tokenize.generate_tokens(StringIO(code_string).readline)      # tokenize the string
    try:
        for _, token_string, _, _, _ in token_generator:
            tokens.append(token_string)
    except tokenize.TokenError:
        # Raised if brackets are opened but never closed and vice versa
        pass
    except IndentationError:
        # Raised if mismatched indentation
        pass

    # Remove empty strings
    tokens = list(filter(None, tokens))
    return tokens


def tokenize_string(code_string):
    """
    Return the given string as a tokenize token generator.
    """
    return tokenize.generate_tokens(StringIO(code_string).readline)


def write_tokens(tokens, path):
    """
    Write a list of tokens to the given path in JSON syntax.

    Parameters
    ----------
    tokens : list of str
        List of tokens to be written.
    path : str
        Path to the resulting JSON file.
    """
    with open(path, 'w') as file:
        json.dump(tokens, file, indent=2)


def get_character_index(code_string, tokens, token_index):
    """
    Returns the index of the first character in the specified token.

    Parameters
    ----------
    code_string : str
        Python source code string.
    tokens : list of str
        List of tokens.
    token_index : int
        The index of the in the list.

    Returns
    -------
    int
        Index of the first character in the token.
    """
    
    # Optimization: If predicted token location is one after the last one in the token sequence (usually for insert after sequence)
    if token_index == (len(tokens)):
        return len(code_string)
    
    # Checks how often the token appears before it appears at the specified position
    token_occurences_before_index = tokens[:token_index].count(tokens[token_index])
    
    return find_nth(code_string, tokens[token_index], token_occurences_before_index + 1)


def get_token_index(code_string, tokens, character_index):
    """
    Returns the index of the token that contains the character at the given index. 

    Parameters
    ----------
    code_string : str
        Python source code string.
    tokens : list of str
        List of tokens.
    character_index : int
        Index of the character.

    Returns
    -------
    """
    
    # Optimization if the token location immediately follows the token sequence (insert after tokens)
    if character_index == len(code_string):
        logging.warning("Character index at the end of code string. Taking last token.")
        return len(tokens)
    
    # Checks how often the token appears before it appears at the specified position
    
    for token_index in range(len(tokens)):
        token_occurences_before_index = tokens[:token_index].count(tokens[token_index])
        calculated_character_index = find_nth(code_string, tokens[token_index], token_occurences_before_index + 1)
        if calculated_character_index == character_index:
            return token_index
        if character_index < calculated_character_index:
            logging.warning("No token found at the specified index. Taking the next closest token.")
            return token_index

    logging.warning("Reached the end of the token without finding the right index. Taking the last token.")
    return len(tokens)


if __name__ == "__main__":
    args = parser.parse_args()

    # extract tokens for the code.
    tokens = to_token_list(args.source)

    # write extracted tokens to file.
    write_tokens(tokens,args.destination)

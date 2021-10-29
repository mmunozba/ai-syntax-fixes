"""
Contains various utility methods for manipulating strings.
"""


def find_nth(sentence, word, n):
    """
    Returns the character index of the nth occurence of the word in the sentence.
    
    Adapted from from https://stackoverflow.com/a/1884277

    Parameters
    ----------
    sentence : str
        Complete sequence of words.
    word : str
        Word to be searched for.
    n : int
        Number of occurence of the token that we want find.

    Returns
    -------
    start : int
        Character were the token occurs.

    """
    start = sentence.find(word)
    while start >= 0 and n > 1:
        start = sentence.find(word, start+len(word))
        n -= 1
    return start


def replace_nth(sentence, word, new_word, n):
    """
    Replaces the nth occurrence of the word in the sentence with a new word.
    Adapted from https://stackoverflow.com/a/35092436

    Parameters
    ----------
    sentence : str
        Complete sequence of words.
    word : str
        Word we want to replace.
    new_word : str
        New word that we want to use to replace the old word.
    n : int
        Number of occurrence of the token that we want to replace.

    Returns
    -------
    str
        New string where the word is replaced.

    """
    find = sentence.find(word)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = sentence.find(word, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return sentence[:find] + new_word + sentence[find+len(word):]
    return sentence


def remove_suffix(input_string, suffix):
    """
    Removes the given suffix from the given string.
    Adapted from Python 3.9: https://docs.python.org/3/library/stdtypes.html#str.removesuffix

    Parameters
    ----------
    input_string : str
        String to remove the suffix from.
    suffix : TYPE
        Suffix to be removed.

    Returns
    -------
    str
        New string without the suffix.

    """
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
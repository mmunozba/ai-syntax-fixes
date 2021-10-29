"""
Module containing the vocabulary class.

"""

SOS_token = 0
EOS_token = 1


class Vocabulary:
    """
    Vocabulary for storing mappings of a vocabulary.
    
    Adapted from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    
    Attributes
    ----------
    word2index : Dict
        Dictionary with an index mapping for each word.
    word2count : Dict
        Dictionary with a word count for each word.
    index2word : Dict
        Dictionary with a word mapping for each index.
    n_words : int
        Number of words in the vocabulary.  
        
    """
    
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS"}
        self.n_words = 1  # Count SOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

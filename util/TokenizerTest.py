"""Runs the tokenizer and checks results for the sample."""
import Tokenizer
import unittest
import os

# Test data
token_list = [
  "def",
  "sum",
  "(",
  "a",
  ",",
  "b",
  ")",
  ":",
  "\n",
  "\t",
  "print",
  "(",
  "a",
  "+",
  "b",
  ")",
  "\n",
  "\n",
  "#sample execution",
  "\n",
  "sum",
  "(",
  "1",
  ",",
  "2",
  ")"
]

code_string_one = "\ndef ID (ID ==ID ,ID ,ID ):\n    pass \n"
code_string_two = "\ndef ID (elif ,ID ,ID ,ID ,ID ):\n    pass \n"
code_string_three = "return def ID (ID ,ID ):\n    ID (ID ,ID ).ID ({\n    LIT :LIT ,\n    LIT :ID \n    })\n"
code_string_four = "\ndef ID (ID ,ID ,ID =LIT and :\n    if ID and ID not in ID .ID :\n        pass \n"
input_path = "test_data/sample.py"
output_path = "test_data/sample_tokens_test.json"
output_correct_path = "test_data/sample_tokens.json"


class TestTokenizer(unittest.TestCase):
    
    def test_to_token_list(self):
        # Read tokens from Python code
        tokens = Tokenizer.to_token_list(input_path)
        self.assertEqual(token_list, tokens)

    def test_write_tokens(self):
        # Write tokens to JSON
        Tokenizer.write_tokens(token_list, output_path)
        
        # Compare the written file with the correct solution
        with open(output_correct_path, 'r') as file:
            correct_tokens_string = file.read()
        with open(output_path, 'r') as file:
            written_tokens_string = file.read()
        self.assertEqual(written_tokens_string, correct_tokens_string)
    
    def test_character_index_from_string(self):
        correct_index = 12
        tokens = Tokenizer.to_token_list_from_string(code_string_one)
        calculated_index = Tokenizer.get_character_index(code_string_one, tokens, 5)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 9
        tokens = Tokenizer.to_token_list_from_string(code_string_two)
        calculated_index = Tokenizer.get_character_index(code_string_two, tokens, 4)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 0
        tokens = Tokenizer.to_token_list_from_string(code_string_three)
        calculated_index = Tokenizer.get_character_index(code_string_three, tokens, 0)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 25
        tokens = Tokenizer.to_token_list_from_string(code_string_four)
        calculated_index = Tokenizer.get_character_index(code_string_four, tokens, 11)
        self.assertEqual(calculated_index, correct_index)
    
    def test_get_token_index(self):
        correct_index = 5
        tokens = Tokenizer.to_token_list_from_string(code_string_one)
        calculated_index = Tokenizer.get_token_index(code_string_one, tokens, 12)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 4
        tokens = Tokenizer.to_token_list_from_string(code_string_two)
        calculated_index = Tokenizer.get_token_index(code_string_two, tokens, 9)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 0
        tokens = Tokenizer.to_token_list_from_string(code_string_three)
        calculated_index = Tokenizer.get_token_index(code_string_three, tokens, 0)
        self.assertEqual(calculated_index, correct_index)
        correct_index = 11
        tokens = Tokenizer.to_token_list_from_string(code_string_four)
        calculated_index = Tokenizer.get_token_index(code_string_four, tokens, 25)
        self.assertEqual(calculated_index, correct_index)
    
    @classmethod
    def tearDownClass(cls):
        # Remove written test file
        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()

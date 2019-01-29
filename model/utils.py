"""
    Created by Mohsen Naghipourfar on 2019-01-29.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def get_char_index(char='A'):
    return ord(char) - 64


def get_sentence_index(sentence="THE QUICK BROWN FOX JUMPS OVER A LAZY DOG"):
    indices = []
    for char in sentence:
        if char != " ":
            indices.append(ord(char) - 64)
    return indices

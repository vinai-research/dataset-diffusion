import pandas as pd
import numpy as np
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from typing import List


def get_token_indices_from_classes(classes: List[str],
                                   prompts: List[str]):
    indices, class_labels, valid = [], [], []
    classes_syn = {}

    for class_id, word in enumerate(classes):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                classes_syn.update({lemma.name(): class_id})

    lemmatizer = WordNetLemmatizer()
    for prompt in prompts:
        prompt = prompt.replace("woman", "person").replace(
            "man", "person").replace("women", "person").replace("men", "person")
        tokens = word_tokenize(prompt)
        normalized_tokens = [lemmatizer.lemmatize(
            token.lower()) for token in tokens]
        curr_indices, curr_labels = [], []
        for i, token in enumerate(normalized_tokens):
            if token in classes_syn:
                curr_indices.append(i + 1)
                curr_labels.append(classes_syn[token])

        indices.append(curr_indices)
        class_labels.append(curr_labels)
        if len(indices) == 0:
            valid.append(False)
        else:
            valid.append(True)
    return indices, class_labels, np.array(valid)


def get_valid_prompts(classes: List[str],
                      prompts: List[dict]):
    indices, class_labels = [], []
    valid_prompts = []
    classes_syn = {}

    for class_id, word in enumerate(classes):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                classes_syn.update({lemma.name(): class_id})

    lemmatizer = WordNetLemmatizer()
    for i, prompt in enumerate(prompts):
        prompt = prompt['caption']
        norm_prompt = prompt.replace("woman", "person").replace(
            "man", "person").replace("women", "person").replace("men", "person")
        tokens = word_tokenize(norm_prompt)
        normalized_tokens = [lemmatizer.lemmatize(
            token.lower()) for token in tokens]
        curr_indices, curr_labels = [], [0]
        for j, token in enumerate(normalized_tokens):
            if token in classes_syn:
                curr_indices.append(j)
                curr_labels.append(classes_syn[token])

        if len(curr_indices) != 0:
            valid_prompts.append(prompts[i])
            indices.append(curr_indices)
            class_labels.append(curr_labels)

    return indices, class_labels, valid_prompts

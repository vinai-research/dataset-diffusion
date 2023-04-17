import pandas as pd
import numpy as np
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from typing import List
DIR = Path(__file__).resolve().parent.parent.parent
PKL = DIR / 'data/voc_captions.pkl'
SAVE_TEXT = DIR / 'data/VOC2012/Captions'


def create_mapping_from_pkl(pickle_path):
    """
    Create corresponding text file that includes the prompt used to generaed the
    image (with the same name of file, but different extension)

    args:
        pickle_path: path to the pickle file that includes a list of dict
                     dict:{
                        "name_image_file.png": <prmopt> ,
                        ...
                     }
    return: none
    """
    cap_and_img = pd.read_pickle(pickle_path)

    for item in cap_and_img:

        img_id, caption = item.values()
        text_id = img_id.split('.')[0] + '.txt'
        with open(SAVE_TEXT / text_id, mode='w', encoding='utf-8') as writer:
            writer.writelines(caption)

    text_list = [item for item in SAVE_TEXT.glob('*.txt')]
    img_path = DIR / 'data/VOC2012/JPEGImages'
    img_list = [item for item in img_path.glob('*.jpg')]
    assert len(text_list) == len(img_list)

# def get_topk_similar_words(model, prompt, base_word, vocab, k=30):
#     text_input = model.tokenizer(
#         [prompt.format(word=base_word)],
#         padding="max_length",
#         max_length=model.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     with torch.no_grad():
#         encoder_output = model.text_encoder(text_input.input_ids.to(model.device))
#     full_prompt_embedding = encoder_output.pooler_output
#     full_prompt_embedding = full_prompt_embedding / full_prompt_embedding.norm(p=2, dim=-1, keepdim=True)

#     prompts = [prompt.format(word=word) for word in vocab]
#     batch_size = 1000
#     all_prompts_embeddings = []
#     for i in tqdm(range(0, len(prompts), batch_size)):
#         curr_prompts = prompts[i:i + batch_size]
#         with torch.no_grad():
#             text_input = model.tokenizer(
#                 curr_prompts,
#                 padding="max_length",
#                 max_length=model.tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             curr_embeddings = model.text_encoder(text_input.input_ids.to(model.device)).pooler_output
#         all_prompts_embeddings.append(curr_embeddings)

#     all_prompts_embeddings = torch.cat(all_prompts_embeddings)
#     all_prompts_embeddings = all_prompts_embeddings / all_prompts_embeddings.norm(p=2, dim=-1, keepdim=True)
#     prompts_similarities = all_prompts_embeddings.matmul(full_prompt_embedding.view(-1, 1))
#     sorted_prompts_similarities = np.flip(prompts_similarities.cpu().numpy().reshape(-1).argsort())

#     print(f"prompt: {prompt}")
#     print(f"initial word: {base_word}")
#     print(f"TOP {k} SIMILAR WORDS:")
#     similar_words = [vocab[index] for index in sorted_prompts_similarities[:k]]
#     print(similar_words)
#     return similar_words

# def get_proxy_words(args, ldm_stable):
#     if len(args.proxy_words) > 0:
#         return [args.object_of_interest] + args.proxy_words
#     vocab = list(json.load(open("vocab.json")).keys())
#     vocab = [word for word in vocab if word.isalpha() and len(word) > 1]
#     filtered_vocab = get_topk_similar_words(ldm_stable, "a photo of a {word}", args.object_of_interest, vocab, k=50)
#     proxy_words = get_topk_similar_words(ldm_stable, args.prompt, args.object_of_interest, filtered_vocab, k=args.number_of_variations)
#     if proxy_words[0] != args.object_of_interest:
#         proxy_words = [args.object_of_interest] + proxy_words

#     return proxy_words

# def get_proxy_prompts(args, ldm_stable):
#     proxy_words = get_proxy_words(args, ldm_stable)
#     prompts = [args.prompt.format(word=args.object_of_interest)]
#     proxy_prompts = [{"word": word, "prompt": args.prompt.format(word=word)} for word in proxy_words]
#     return proxy_words, prompts, proxy_prompts


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
                      prompts: List[str]):
    indices, class_labels = [], []
    valid_prompts = []
    classes_syn = {}

    for class_id, word in enumerate(classes):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                classes_syn.update({lemma.name(): class_id})

    lemmatizer = WordNetLemmatizer()
    for i, prompt in enumerate(prompts):
        norm_prompt = prompt.replace("woman", "person").replace(
            "man", "person").replace("women", "person").replace("men", "person")
        tokens = word_tokenize(norm_prompt)
        normalized_tokens = [lemmatizer.lemmatize(
            token.lower()) for token in tokens]
        curr_indices, curr_labels = [], [0]
        for j, token in enumerate(normalized_tokens):
            if token in classes_syn:
                curr_indices.append(j + 1)
                curr_labels.append(classes_syn[token])

        if len(curr_indices) != 0:
            valid_prompts.append(i)
            indices.append(curr_indices)
            class_labels.append(curr_labels)

    return indices, class_labels, valid_prompts

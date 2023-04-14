import torch
import torch.nn.functional as F
from typing import List 
import numpy as np
from PIL import Image
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .crf import dense_crf, multi_class_dense_crf
# from .pipeline_stable_diffusion_with_attention import AttentionStore


def ensemble_normalized_cross_attention(attention_store,
                                        token_indices: List[int]):
    """
    A_{s, j} = 1/(S*T) * \sum (A_{s,t} / max(A_{s,t}))
    """
    
    max_resolution  = 512
    attention_store_res = dict()
    #   Get the attention maps of each token at different scale
    attention_maps = attention_store.get_average_attention()

    for k, attention_map_list in attention_maps.items():
        list_cross_attention    = []               #   Use to store resized cross attention map (resize to the max resolution)
        for attention_map in attention_map_list:
            for i in range(len(attention_map)):
                token_attention = attention_map[range(len(attention_map)), :, :, :, token_indices]
                resolution = token_attention.shape[-1]

                if resolution < max_resolution:
                    token_attention = F.interpolate(token_attention, size=max_resolution, mode = 'bilinear')

                list_cross_attention.append(token_attention)     
                attention_store_res[k] = torch.cat(list_cross_attention, dim=1).mean(dim=1)

    average_attention_map = (attention_store_res['down_cross'] + \
        attention_store_res['mid_cross'] + attention_store_res['up_cross']) / 3
    return average_attention_map


def generate_mask_dense_crf(attention_store,
                            token_indices,
                            gen_images,
                            prompts,
                            dense_crf_threshold=0.5,
                            labels=None):
    max_resolution  = 512
    attention_store_res = dict()
    #   Get the attention maps of each token at different scale
    attention_maps = attention_store.get_average_attention()
    masks = []
    if prompts is not None and isinstance(prompts, str):
        batch_size = 1
        prompts = [prompts]
    elif prompts is not None and isinstance(prompts, list):
        batch_size = len(prompts)
        
    if (token_indices is not None) and (not isinstance(token_indices[0], list)):
        token_indices = [token_indices]
        
    for i in range(batch_size):
        for k, attention_map_list in attention_maps.items():
            list_cross_attention    = []               #   Use to store resized cross attention map (resize to the max resolution)
            assert len(attention_map_list[0]) == len(token_indices) == batch_size
            for attention_map in attention_map_list:
                token_attention = attention_map[i, :, :, :, token_indices[i]].permute(3, 0, 1, 2)
                resolution = token_attention.shape[-1]

                if resolution < max_resolution:
                    token_attention = F.interpolate(token_attention, size=max_resolution, mode='bilinear')

                list_cross_attention.append(token_attention)     
                attention_store_res[k] = torch.cat(list_cross_attention, dim=1).mean(dim=1)

        average_attention_map = (attention_store_res['down_cross'] + \
            attention_store_res['mid_cross'] + attention_store_res['up_cross']) / 3
        average_attention_map = average_attention_map.cpu().numpy()
        average_attention_map = (average_attention_map - average_attention_map.min((1, 2), keepdims=True)) / \
                            (average_attention_map.max((1, 2), keepdims=True) - average_attention_map.min((1, 2), keepdims=True))
        
        mask = multi_class_dense_crf(attention=average_attention_map, image=gen_images[i], threshold=dense_crf_threshold)
        mask = np.argmax(mask, axis=0)
        if labels is not None:
            label = np.array(labels[i], dtype=np.uint8)
            mask = label[mask]
        masks.append(mask)
    return masks


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
        prompt = prompt.replace("woman", "person").replace("man", "person").replace("women", "person").replace("men", "person")
        tokens = word_tokenize(prompt)
        normalized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
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
    for prompt in prompts:
        norm_prompt = prompt.replace("woman", "person").replace("man", "person").replace("women", "person").replace("men", "person")
        tokens = word_tokenize(norm_prompt)
        normalized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        curr_indices, curr_labels = [], [0]
        for i, token in enumerate(normalized_tokens):
            if token in classes_syn:
                curr_indices.append(i + 1)
                curr_labels.append(classes_syn[token])
                
        if len(curr_indices) != 0:
            valid_prompts.append(prompt)
            indices.append(curr_indices)
            class_labels.append(curr_labels)
            
    return indices, class_labels, valid_prompts


def image_grid(imgs, rows=2, cols=2):                                                                                                                                                                                                         
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))                                                                                                                                                                                            
    return grid

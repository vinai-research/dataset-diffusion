import torch
import numpy as np
import abc
from typing import Optional, Union, Tuple, Dict
import sys
from .seq_aligner import get_replacement_mapper
from diffusers.models.cross_attention import CrossAttention
import math

class AttendExciteCrossAttnProcessor:

    # def __init__(self, attnstore, place_in_unet):
    #     super().__init__()
    #     self.attnstore = attnstore
    #     self.place_in_unet = place_in_unet

    # def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
    #     batch_size, sequence_length, _ = hidden_states.shape
    #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

    #     query = attn.to_q(hidden_states)

    #     is_cross = encoder_hidden_states is not None
    #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    #     key = attn.to_k(encoder_hidden_states)
    #     value = attn.to_v(encoder_hidden_states)

    #     query = attn.head_to_batch_dim(query)
    #     key = attn.head_to_batch_dim(key)
    #     value = attn.head_to_batch_dim(value)

    #     attention_probs = attn.get_attention_scores(query, key, attention_mask)

    #     self.attnstore(attention_probs, is_cross, self.place_in_unet)

    #     hidden_states = torch.bmm(attention_probs, value)
    #     hidden_states = attn.batch_to_head_dim(hidden_states)

    #     # linear proj
    #     hidden_states = attn.to_out[0](hidden_states)
    #     # dropout
    #     hidden_states = attn.to_out[1](hidden_states)

    #     return hidden_states

    def __init__(self, attn_store, place_in_unet):
        self.attn_store = attn_store
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # attention_probs = attention_probs.reshape(batch_size, attn.heads, -1, sequence_length).mean(dim=1)
        self.attn_store(attention_probs, attn.heads, is_cross=is_cross, place_in_unet=self.place_in_unet)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


# class AttentionStore(AttentionControl):
class AttentionStore:
    # @staticmethod
    # def get_empty_store():
    #     return {"down_cross": [], "mid_cross": [], "up_cross": [],
    #             "down_self": [], "mid_self": [], "up_self": []}

    # def forward(self, attn, is_cross: bool, place_in_unet: str):
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #     if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
    #         self.step_store[key].append(attn)
    #     return attn

    # def between_steps(self):
    #     self.attention_store = self.step_store
    #     if self.save_global_store:
    #         with torch.no_grad():
    #             if len(self.global_store) == 0:
    #                 self.global_store = self.step_store
    #             else:
    #                 for key in self.global_store:
    #                     for i in range(len(self.global_store[key])):
    #                         self.global_store[key][i] += self.step_store[key][i].detach()
    #     self.step_store = self.get_empty_store()
    #     self.step_store = self.get_empty_store()

    # def get_average_attention(self):
    #     average_attention = self.attention_store
    #     return average_attention

    # def get_average_global_attention(self):
    #     average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
    #                          self.attention_store}
    #     return average_attention

    # def reset(self):
    #     super(AttentionStore, self).reset()
    #     self.step_store = self.get_empty_store()
    #     self.attention_store = {}
    #     self.global_store = {}

    # def __init__(self, save_global_store=False):
    #     '''
    #     Initialize an empty AttentionStore
    #     :param step_index: used to visualize only a specific step in the diffusion process
    #     '''
    #     super(AttentionStore, self).__init__()
    #     print(save_global_store)
    #     self.save_global_store = save_global_store
    #     self.step_store = self.get_empty_store()
    #     self.attention_store = {}
    #     self.global_store = {}
    #     self.curr_step_index = 0

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, n_heads: int, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        # attn: (batch_size * heads, res*res, n_tokens) --> (batch_size, heads, res, res, n_tokens)
        spatial_res = int(math.sqrt(attn.shape[1]))
        
        if self.do_classifier_free_guidance:
            attn = attn.reshape(2, -1, n_heads, spatial_res, spatial_res, attn.shape[-1]) # [2, batch_size, n_heads, res, res, n_tokens]
            attn = attn.transpose(0, 1).flatten(1, 2)
        else:
            attn = attn.reshape(-1, n_heads, spatial_res, spatial_res, attn.shape[-1])
        #     attn = attn.transpose(0, 2).flatten(1, 2)
        # if self.do_classifier_free_guidance:
        #     attn, _ = attn.chunk(2)
        # if self.do_classifier_free_guidance:
        #     _, attn = attn.chunk(2)
        self.attn_store[key].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.num_steps += 1
            if self.save_global_store:
                if len(self.global_store) == 0:
                    self.global_store = self.attn_store
                for key in self.global_store:
                    for i in range(len(self.attn_store[key])):
                        self.global_store[key][i] += self.attn_store[key][i]
            else:
                self.global_store = self.attn_store
            self.attn_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.num_steps for item in self.global_store[key]] for key in
                             self.global_store}
        return average_attention

    def __init__(self, do_classifier_free_guidance=True, save_global_store=True):
        self.attn_store = self.get_empty_store()
        self.global_store = {}
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.num_steps = 0
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.save_global_store = save_global_store


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words

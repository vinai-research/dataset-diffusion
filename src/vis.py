from helper.attention_helper import show_cross_attention, show_self_attention_comp
import numpy as np
import pickle
from pathlib import Path
from configs import PathConfig

path_config = PathConfig()
Base = path_config.BASE
save_self_folder = path_config.SAVE / f'{Base}/AttentionVis'
save_cross_folder = path_config.SAVE / f'{Base}/CrossVis'

save_self_folder.mkdir(parents=True, exist_ok=True)
save_cross_folder.mkdir(parents=True, exist_ok=True)


def read_attention_pkl(attn_folder: Path, is_cross: bool):

    pkl_list = [item for item in attn_folder.glob('*.pkl')]

    for idx, item in enumerate(pkl_list):
        res = None
        with open(item, mode='rb') as reader:
            res = pickle.load(reader)
        reader.close()
        breakpoint()
        filename, ext = item.split('/')[-1].split('.')

        # cross_attn =

read_attention_pkl(path_config.BASEDIR / 'save/VOC2012/Attention', False)

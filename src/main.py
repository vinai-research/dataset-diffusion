import numpy as np
import json
import torch
from configs import *
from model import SDMaskWrapper
from generation.pipeline_attend_and_excite import AttendAndExcitePipeline
from diffusers.pipelines import StableDiffusionPipeline
import pickle
import matplotlib.pyplot as plt
from helper.dataset import CaptionDataset, DictCaptionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper.attention_helper import show_cross_attention, show_self_attention_comp
from helper.captions import get_valid_prompts


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_stable_diffusion(model_path):
    """
    Using the attend-and-excite to maximize the corr between addressed token
    with some patches in images.
    """
    model = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16).to(DEVICE)
    model.enable_attention_slicing()
    return model


def get_prompt(fn):
    prompt = None
    with open(fn, mode='r', encoding='utf-8') as reader:
        prompt = reader.readlines()[0].strip('\n')
    return prompt


def get_filename_extension(path):
    return str(path).split('/')[-1].split('.')


def main():
    path_config = PathConfig()
    path_config.pretrained = path_config.PRETRAINED
    sd_run_config = GeneratorConfig()
    segmentor_config = SegmentorConfig()
    dcrf_config = DenseCRFConfig()
    Base = path_config.BASE
    #   Setup text folder for getting the input prompt
    text_folder = [item for item in path_config.TEXTDIR.glob('*.txt')]

    #   Configuration for saving
    save_img_folder = path_config.SAVE / f'{Base}/JPEGImages'
    save_attn_folder = path_config.SAVE / f'{Base}/Attention'
    save_mask_folder = path_config.SAVE / f'{Base}/MaskPkl'
    save_mask_figures = path_config.SAVE / f'{Base}/MaskFig'
    save_dcrf_folder = path_config.SAVE / f'{Base}/DcrfPkl'
    save_dcrf_figures = path_config.SAVE / f'{Base}/DcrfFig'
    save_cross_figs = path_config.SAVE / f'{Base}/CrossFig'
    save_self_figs = path_config.SAVE / f'{Base}/SelfFig'
    #   Create subfolder
    save_img_folder.mkdir(parents=True, exist_ok=True)
    save_mask_folder.mkdir(parents=True, exist_ok=True)
    save_attn_folder.mkdir(parents=True, exist_ok=True)
    save_mask_figures.mkdir(parents=True, exist_ok=True)
    save_cross_figs.mkdir(parents=True, exist_ok=True)
    save_self_figs.mkdir(parents=True, exist_ok=True)
    # save_dcrf_figures.mkdir(parents = True, exist_ok=True)
    print(f'{path_config}\n{sd_run_config}\n{segmentor_config}')

    model = get_stable_diffusion(path_config.pretrained)
    model_config = None
    # breakpoint()
    model = SDMaskWrapper(
        path_config,
        segmentor_config,
        dcrf_config,
        model,
        model_config
    )

    #   Prepare data
    classes=('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor')

    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]]

    with open(path_config.DATADIR) as f:
        data = json.load(f)
    prompts = [prompt['caption'] for prompt in data]
    indices, class_labels, valid_prompts = get_valid_prompts(classes, prompts)
    data = [data[i] for i in valid_prompts]
    caption_dataset = DictCaptionDataset(data)
    caption_dataloader = DataLoader(
        caption_dataset, batch_size=2, shuffle=False)
    cluster_mask_result = dict()
    dcrf_results = dict()

    count = 0
    NUM_SAMPLES = 50
    for idx, batch in tqdm(enumerate(caption_dataloader)):
        file_name, text = batch

        for idx, item in enumerate(file_name):
            print(f'Filename: {item} : {text[idx]}')
        if not isinstance(text, list):
            text = list(text)

        # indices  = [[0,1], [0, 1 ,2]]
        indices = [[2, 3] for i in range(len(text))]
        assert len(indices) == len(
            text), "Number of list of index must be equal to the number of prompts"
        image, cluster_512, attn_result = model(
            text, indices, sd_run_config)

        self_attn = attn_result['self']
        cross_attn = attn_result['cross']
        breakpoint()
        for idx, filename in enumerate(file_name):
            image[0][idx].save(save_img_folder / f'{filename}.jpg')
            # cluster_512 = cluster_512.detach().cpu().numpy()
            cluster_mask_result[filename] = cluster_512[idx]

            # dcrf_results[filename] = dcrf_mask
            plt.imshow(cluster_512[idx])
            plt.axis('off')
            plt.savefig(f'{save_mask_figures}/{filename}.png')

            self_attn_vis = self_attn[idx]
            cross_attn_vis = cross_attn[idx]

            images_cross_rel = show_cross_attention(cross_attn_vis.mean(
                dim=0), model.controller, 16, ['up', 'down'], text, indices[idx], 0, image[0][idx])
            images_cross_rel.save(f'{save_cross_figs}/{filename}.png')

            images_self_rel = show_self_attention_comp(
                self_attn_vis.mean(dim=0), model.controller, 32, ['up', 'down'])
            images_self_rel.save(f'{save_self_figs}/{filename}.png')

            with open(f'{save_attn_folder}/{filename}.pkl', mode='wb') as writer:
                self_cross_per_file = {
                    'self': self_attn_vis, 'cross': cross_attn_vis}
                pickle.dump(self_cross_per_file, writer)
            writer.close()

            with open(f'{save_mask_folder}/{filename}.pkl', mode='wb') as writer:
                pickle.dump(cluster_mask_result, writer)
            writer.close()

            # with open(f'{save_dcrf_folder}/{filename}.pkl', mode = 'wb') as writer:
            #     pickle.dump(dcrf_results, writer)
            # writer.close()

        if count > NUM_SAMPLES:
            break
        count += len(text)


if __name__ == '__main__':
    main()

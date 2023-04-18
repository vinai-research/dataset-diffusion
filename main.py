from diffusers.pipelines import StableDiffusionPipeline
from src.captions import get_valid_prompts
from src.controller import AttentionStore, register_attention_control
from src.attention_based_segmentation import Segmentor
from src.attention_utils import show_cross_attention, show_self_attention_comp
import matplotlib.pyplot as plt 
from PIL import Image
import json
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--save-attention', action='store_true', help='whether save the attention maps')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    pipe = StableDiffusionPipeline.from_pretrained("/vinai/truongvt/checkpoints/stable-diffusion-v1-5/", torch_dtype=torch.float16).to(device)
    pipe.enable_attention_slicing()

    controller = AttentionStore()
    register_attention_control(pipe, controller)

    segmentor = Segmentor(controller)

    ## From https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/voc.py
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

    ######################################################################################

    palette = [value for color in palette for value in color]

    generator = torch.Generator(device=device).manual_seed(0)
    with open("data/voc_captions.json") as f:
        prompts = json.load(f)
    indices, labels, valid_prompts = get_valid_prompts(classes, prompts)
    print(f"Total: {len(valid_prompts)}")
    batch_size = args.batch_size
    curr_index = 0
    start_index = max(0, args.start)
    if args.end == -1:
        end_index = len(valid_prompts)
    else:
        end_index = min(len(valid_prompts), args.end)
    for i in range(start_index, end_index, batch_size):
        batch = valid_prompts[i:i+batch_size]
        batch_filenames = [x['filename'] for x in batch]
        batch_prompts = [x['caption'] for x in batch]
        batch_indices = indices[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        out = pipe(batch_prompts, num_inference_steps=100, generator=generator, output_type="numpy")
        clusters, out_crfs, self_attention, cross_attention = segmentor(out.images, controller, batch_indices, batch_labels)
        
        for j in range(len(out.images)):
            base_filename = batch_filenames[j].split(".")[0]
            image = (out.images[j] * 255.).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"figs/image/{base_filename}.jpg", image)
            
            mask = Image.fromarray(out_crfs[j].astype(np.uint8)).convert("P")
            mask.putpalette(palette)
            mask.save(f"figs/mask_crf/{base_filename}.png")

            if args.save_attention:
                mask = Image.fromarray(clusters[j]).convert("P")
                mask.putpalette(palette)
                mask.save(f"figs/cluster/{base_filename}.png")        
                
                images_self_rel = show_self_attention_comp(self_attention[j], 32, ['up', 'down'], select=j) 
                images_self_rel.save(f"figs/self_attention/{base_filename}.png")
                
                images_cross_rel = show_cross_attention(cross_attention[j], 32, ['up', 'down'], batch_prompts, pipe.tokenizer, select=j) 
                images_cross_rel.save(f"figs/cross_attention/{base_filename}.png")

        controller.reset()

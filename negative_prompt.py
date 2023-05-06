from diffusers.pipelines import StableDiffusionPipeline
from src.captions import get_valid_prompts
from src.controller import AttentionStore, register_attention_control, aggregate_attention
from src.attention_based_segmentation import Segmentor
from src.attention_utils import show_cross_attention, show_self_attention_comp, show_cross_attention_relevance
import matplotlib.pyplot as plt 
from PIL import Image
import json
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch

CAPTION = 'a man is riding a horse'
NEGATIVE = 'a horse'
if __name__ == '__main__':
    """Prepare models for generation and segmentation"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = StableDiffusionPipeline.from_pretrained("pretrained/stable-diffusion-2-1-base", torch_dtype=torch.float16).to(device)
    pipe.enable_attention_slicing()
    controller = AttentionStore()

    use_negative =True
    register_attention_control(pipe, controller)
    segmentor = Segmentor(controller)
    generator = torch.Generator(device=device).manual_seed(0)
    
    base_filename = "test"
    prompts = [
        {
            'filename': 'test.jpg',
            'caption': CAPTION,
            'negative': NEGATIVE,
        }
    ]

    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]]
    classes=('aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor')
    indices, labels, valid_prompts = get_valid_prompts(classes, prompts)

    negative_prompt = [prompts[0]['negative']]
    # negative_prompt = None
    batch_prompts = [prompts[0]['caption']]
    
    out = pipe(batch_prompts, num_inference_steps=50, generator=generator, output_type="numpy", negative_prompt = negative_prompt)
    self_attention = None
    cross_attention = None
    cross_attention_neg = None

    if negative_prompt is None:
        self_attention = aggregate_attention(controller, 32, ['up', 'down'], is_cross=False, select = 0)
        cross_attention = aggregate_attention(controller, 16, ['up', 'down'], is_cross = True, select = 0)
    # clusters, out_crfs, self_attention, cross_attention = segmentor(out.images, controller, indices[:], labels[:])
    else:

        self_attention, self_attention_neg = aggregate_attention(controller, 32, ['up', 'down'], is_cross=False, select = 0, negative_prompt=negative_prompt)
        cross_attention, cross_attention_neg = aggregate_attention(controller, 16, ['up', 'down'], is_cross = True, select = 0, negative_prompt=negative_prompt)

    
    # image_cross = show_cross_attention(cross_attention, 16, ['up', 'down'], batch_prompts, pipe.tokenizer, out.images[0], select = 0)
    image_cross = show_cross_attention_relevance(cross_attention, out.images[0], pipe.tokenizer, batch_prompts, select = 0)
    image_cross.save('test_cross_attn_with_neg.png')
    if len(negative_prompt) > 0 :
        image_cross_neg = show_cross_attention_relevance(cross_attention_neg, out.images[0], pipe.tokenizer, negative_prompt, select = 0)
        image_cross_neg.save('test_cross_attn_neg.png')

    
    

    # breakpoint()
    # if cross_attention_neg is not None:
    #     image_cross = show_cross_attention(cross_attention_neg, 16, ['up', 'down'], batch_prompts, pipe.tokenizer, select = 0)
    # images_self_rel = show_self_attention_comp(self_attention[j], 32, ['up', 'down'], select=j) 
    # images_self_rel.save(f"figs/self_attention/{base_filename}.png")
    
    # images_cross_rel = show_cross_attention(cross_attention[j], 32, ['up', 'down'], batch_prompts, pipe.tokenizer, select=j) 
    # images_cross_rel.save(f"figs/cross_attention/{base_filename}.png")
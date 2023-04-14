from src.model.pipeline_stable_diffusion_with_attention import StableDiffusionPipelineWithAttention
from src.model.utils import get_token_indices_from_classes, ensemble_normalized_cross_attention, get_valid_prompts, generate_mask_dense_crf
from src.model.crf import dense_crf, multi_class_dense_crf
import matplotlib.pyplot as plt 
from PIL import Image
import json
import cv2
from tqdm import tqdm
import numpy as np
import torch

pipe = StableDiffusionPipelineWithAttention.from_pretrained("/vinai/truongvt/checkpoints/stable-diffusion-2-1-base/", torch_dtype=torch.float16)
pipe = pipe.to("cuda:2")
pipe.enable_attention_slicing()

with open("voc_captions.json") as f:
    data = json.load(f)
prompts = [prompt['caption'] for prompt in data]


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

generator = torch.Generator(device="cuda:2").manual_seed(0)
indices, labels, valid_prompts = get_valid_prompts(classes, prompts)

batch_size = 3
curr_index = 0
for i in tqdm(range(0, len(valid_prompts), batch_size)):
    out = pipe(valid_prompts[i:i+batch_size], num_inference_steps=50, generator=generator, output_type="numpy")
    masks = generate_mask_dense_crf(pipe.attention_store, indices[i:i+batch_size], \
        out.images, valid_prompts[i:i+batch_size], dense_crf_threshold=0.45, labels=labels[i:i+batch_size])
    
    for image, mask in zip(out.images, masks):
        image = (image * 255.).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(mask).convert("P")
        mask.putpalette(palette)
        mask.save(f"figs/segmentation/{curr_index}.png")
        cv2.imwrite(f"figs/image/{curr_index}.jpg", image)
        curr_index += 1
        
    if i % 5 == 0:
        print(f"Finish {i} images")

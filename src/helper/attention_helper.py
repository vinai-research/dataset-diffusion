import cv2
import torch
import numpy as np
from typing import Tuple, List
from cv2 import putText, getTextSize, FONT_HERSHEY_SIMPLEX
from PIL import Image
import sys
from .attention_store import AttentionStore
import os
from einops import rearrange
import math

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts):
    """Return the aggragation of attention maps across down-mid-up stream  
        at the specific resolution

        return:
            attention_maps: [batch, n_heads, res, res, dim]
    """
    out = []
    attention_maps = attention_store.get_average_attention()
 
    # breakpoint()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            _, item_cond_text = item.chunk(2)           # item_cond_text: [batch, dims, res, res, n_tokens/ res**2]
            # item = item.mean(dim = 1)
            if item_cond_text.shape[2]**2 == num_pixels: 
                cross_maps = item_cond_text.reshape(1, len(prompts), item.shape[1], res, res, item.shape[-1]) 
                # cross_maps   = item.sum(dim = 0) / item.shape[0]            
                out.append(cross_maps)

    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    #   out: [batch, dims, res, res, n_tokens / res**2]
    return out.cpu()

def show_cross_attention(cross_attention, attention_store: AttentionStore, 
                         res: int, from_where: List[str], prompts, token_indices, select: int,
                         orig_image):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decod
    if cross_attention is None:
        attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    else:
        attention_maps = cross_attention
    images = []
    for i in token_indices:
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, orig_image)
        # image = 255 * image / image.max()
        # image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        # image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return view_images(np.stack(images, axis=0))


def show_self_attention_comp(self_attn, attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=12, select: int = 0):

    if self_attn is None:
        attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
            (res ** 2, res ** 2))

    
    attention_maps = self_attn
    es, _, dim = attention_maps.shape
    attention_maps = attention_maps.detach().cpu().numpy()
    attention_maps = attention_maps.astype('float32')
    attention_maps = attention_maps.reshape((res ** 2, res**2))  
    # breakpoint() 
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
     
    images = []
    for i in range(max_com):  
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    return view_images(np.concatenate(images, axis=1))

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    # display(pil_img)
    return pil_img

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

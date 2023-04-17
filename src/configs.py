from dataclasses import dataclass 
from pathlib import Path
from typing import List

@dataclass
class GeneratorConfig:
    
    attention_res           = 32
    height                  = None
    width                   = None
    num_inference_steps     = 100
    guidance_scale          = 7.5
    negative_prompt         = None
    num_images_per_prompt   = 1
    eta                     = 0.0
    # generator               = None
    latents                 = None
    prompt_embeds           = None
    negative_prompt_embeds  = None
    output_type             = 'pil'
    return_dict             = True
    callback                = None
    callback_steps          = 1
    cross_attention_kwargs  = None
    max_iter_to_alter       = 10
    thresholds              = {0: 0.05, 10: 0.5, 20: 0.8}
    scale_factor            = 20
    scale_range             = (1., 0.5)
    smooth_attentions       = False
    sigma                   = 0.5
    kernel_size             = 3 
    run_standard_sd         = False
    seed                    = 1123

@dataclass
class SegmentorConfig:
    seed:   int                         = 1123

    #   0 - Kmeans, 1 - DBSCAN, 2 - Optics
    method: int                             = 0
    num_segments: int                       = 5
    bg_threshold: float = 0.45
    resolution: int                         = 32
    bg_nouns                          = []

@dataclass
class PathConfig:
    BASE        = 'VOC2012_1'
    BASEDIR     = Path(__file__).resolve().parent.parent
    DATADIR     = BASEDIR / 'data'
    PRETRAINED  = BASEDIR / 'pretrained'
    SAVE        = BASEDIR / 'save'
    TEXTDIR     = DATADIR / 'VOC2012/Captions'

@dataclass 
class DenseCRFConfig:
    MAX_ITER = 10
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3
    threshold  = 0.45
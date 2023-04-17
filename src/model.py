import sys
# sys.path.append('.')
# sys.path.append('..')

from typing import Optional, List
import numpy as np 
import torch
import torch.nn.functional as F
from cv2 import dilate
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

from helper.attention_helper import aggregate_attention, show_cross_attention
from helper.attention_store  import AttentionStore, AttendExciteCrossAttnProcessor

from diffusers import StableDiffusionPipeline
from segmentation.attention_based_segmentation import Segmentor

from configs import GeneratorConfig, SegmentorConfig, PathConfig, DenseCRFConfig

path_config = PathConfig()

class SDMaskWrapper:
    """
    args:
        args:
        segmentor_config:
        dcrf_config:
        model:
        model_config:
        prompt_mixing:
        generator: (default = None)
    
    attributes:
        args:
        model:
        controller:
        prompt_mixing
        device
        generator_seed
        heigh
        width
        diff_step
    """

    def __init__(self,  args,
                        segmentor_config: SegmentorConfig, 
                        dcrf_config: DenseCRFConfig,
                        model,
                        model_config: GeneratorConfig, 
                        prompt_mixing = None, generator = None):
        
        self.args               = args
        self.model              = model 
    
        self.controller         = AttentionStore(False)
        self.generator_config   = model_config
        self.dcrf_config        = dcrf_config
        if self.controller is None:
            raise ValueError("Attention store need to be initialized")
        
        self.prompt_mixing      = prompt_mixing
        self.device             = self.model.device
        self.generator_seed     = generator
        self.height             = 512
        self.width              = 512

        self.diff_step          = 0
        self.register_attention_control()
        self.segmentor          = Segmentor(
            method              = segmentor_config.method,
            num_segments        = segmentor_config.num_segments,
            background_segment_threshold = segmentor_config.bg_threshold,
            res                 = segmentor_config.resolution,
            background_nouns    = []
        ) 
        
    def run_on_prompt(self, prompt: List[str], token_indices: List[int], run_config):

        gen_seed = torch.Generator(self.device).manual_seed(run_config.seed)
        output, decoded_latents = self.model(prompt, self.controller, token_indices, run_config, gen_seed)
        image  = output.images
        return image, decoded_latents

    def register_attention_control(self):

        attn_procs = {}
        cross_att_count = 0
        for name in self.model.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.model.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.model.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.model.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.model.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attn_store=self.controller, place_in_unet=place_in_unet
            )

        self.model.unet.set_attn_processor(attn_procs)
        self.controller.num_att_layers = cross_att_count

    @staticmethod 
    def multi_class_dense_crf(image: np.array, attention: np.array, 
                              threshold: float, POS_XY_STD: float, POS_W: float, Bi_XY_STD: float,
                              Bi_RGB_STD: float, Bi_W: float, MAX_ITER: int):
        if image.dtype == np.float32:
            image = (image * 255.).astype(np.uint8)
        num_classes = attention.shape[0] + 1
        image = np.ascontiguousarray(image)
        breakpoint()
        mask = np.full_like(attention[0], threshold)
        labels = np.zeros_like(attention[0], dtype=np.int32)
        for i in range(num_classes-1):
            labels[attention[i] > mask] = i + 1
            mask = np.maximum(mask, attention[i])
        
        h = labels.shape[0]
        w = labels.shape[1]
        U = utils.unary_from_labels(labels, num_classes, 0.7, zero_unsure=False)
        U = np.ascontiguousarray(U)

        d = dcrf.DenseCRF2D(w, h, num_classes)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
        d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

        Q = d.inference(MAX_ITER)
        Q = np.array(Q).reshape((num_classes, h, w))
        return Q 
        
    def generate_mask_dense_crf(self,
                                token_indices,
                                gen_images,
                                prompts,
                                dense_crf_threshold=0.5,
                                labels=None):
        max_resolution  = 512
        attention_store_res = dict()
        #   Get the attention maps of each token at different scale
        attention_maps = self.controller.get_average_attention()
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
                breakpoint() 
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
            
            # mask = self.multi_class_dense_crf(attention=average_attention_map, image=gen_images[i], threshold=dense_crf_threshold)

            mask   = self.multi_class_dense_crf(gen_images, average_attention_map, threshold = dense_crf_threshold,
                                            POS_XY_STD = self.dcrf_config.POS_XY_STD,
                                            POS_W = self.dcrf_config.POS_W,
                                            Bi_XY_STD = self.dcrf_config.Bi_XY_STD,
                                            Bi_RGB_STD = self.dcrf_config.Bi_RGB_STD,
                                            Bi_W = self.dcrf_config.Bi_W,
                                            MAX_ITER = self.dcrf_config.MAX_ITER)
            mask = np.argmax(mask, axis=0)
            if labels is not None:
                label = np.array(labels[i], dtype=np.uint8)
                mask = label[mask]
            masks.append(mask)
        return masks

    def refined_selfattn_dcrf(self, gen_images, cluster):
        """
        args:
            gen_images: generated images in numpy array: shape [batch, 512, 512, 3]
            pseudo_mask: cluster of self attention: shape [batch, 512, 512]
            dcrf_threshold: float
            dcrf_config: DenseCRFConfig: can be founded in /configs.py
        """
        masks = []

        for idx, image in enumerate(gen_images):
            pseudo_mask = cluster[idx]
            mask   = self.multi_class_dense_crf(image, pseudo_mask, threshold = self.dcrf_config.threshold,
                                            POS_XY_STD = self.dcrf_config.POS_XY_STD,
                                            POS_W = self.dcrf_config.POS_W,
                                            Bi_XY_STD = self.dcrf_config.Bi_XY_STD,
                                            Bi_RGB_STD = self.dcrf_config.Bi_RGB_STD,
                                            Bi_W = self.dcrf_config.Bi_W,
                                            MAX_ITER = self.dcrf_config.MAX_ITER)
            masks.append(mask)
        breakpoint() 
        pass
    
    @staticmethod
    def get_attention(attn_store: AttentionStore, resolution: int, is_cross: bool, prompts):
        """
        args:
            attn_store: AttentionStore object to extract the attention
            is_cross: return the cross attention if true, otherwise return the self attention
        return:
            {self/cross}_attention: [batch, n_heads, res, res, dim]
        """
        attention = aggregate_attention(
            attn_store, 
            res = resolution,
            from_where = ['up', 'down'],
            is_cross = is_cross,
            select = len(prompts) -1,
            prompts = prompts
        )

        return attention


    @torch.no_grad()
    def __call__(self, 
                prompt: List[str],
                token_indices: List[int],
                GEN_CONFIG: GeneratorConfig):

        attn_result = dict()
        #   Generate image given the prompt
        image, decoded_latents          = self.run_on_prompt(prompt, token_indices, GEN_CONFIG)
        
        self_attention  = self.get_attention(self.controller, 32, is_cross = False, prompts = prompt)
        cross_attention = self.get_attention(self.controller, 16, is_cross = True, prompts = prompt)

        """
            Generating pseudo masked by clustering the self-attention and 
            then refine to the final masked with the detected objects noun in
            the sentence inputs
        """
        cluster         = self.segmentor(prompts = prompt, self_attention=self_attention)
        cluster2noun    = self.segmentor.cluster2noun(cluster, cross_attention)

        #   Cluster: np.array: shape: [batch, 32, 32]
        B, res, _ = cluster.shape
        image_size = 512
        cluster         = torch.Tensor(cluster).view(1, 1, B, res, res) 
        cluster_512     = torch.nn.functional.interpolate(cluster, (B, image_size, image_size), mode = 'nearest')
        cluster_512     = cluster_512.squeeze(0).squeeze(0) 

        # if cluster_512.shape != (512, 512):
        #     raise ValueError(f'Expected size of clustering self attention = {512} after interpolation')
        attn_result['self']     = self_attention
        attn_result['cross']    = cross_attention

        # self.refined_selfattn_dcrf(decoded_latents, cluster_512.detach().cpu().numpy())
        # """
        #     Generating pseudo mask by applying dense conditional random field
        #     with the configurations defined in DenseCRFConfig class.
        # """ 
        # # densecrf_mask   = self.generate_mask_dense_crf(token_indices = [token_indices], gen_images = image, prompts=prompt)
        # densecrf_mask     = None
        # # breakpoint()

        # return image, cluster_512, attn_result, densecrf_mask
        return image, decoded_latents, cluster_512, attn_result
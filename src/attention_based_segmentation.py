from typing import List
import nltk
from sklearn.cluster import KMeans
import numpy as np
import cv2
from scipy import interpolate
from PIL import Image

from .attention_utils import aggregate_attention
from .crf import multi_class_dense_crf
from .controller import AttentionControl


class Segmentor:

    def __init__(self, controller, num_segments=5, background_segment_threshold=0.3, background_nouns=[]):
        self.controller = controller
        self.num_segments = num_segments
        self.background_segment_threshold = background_segment_threshold
        self.background_nouns = background_nouns

    def __call__(self, images: np.array, controller: AttentionControl, indices: List[int], labels: List[int]):
        clusters, out_crfs = [], []
        image_res = images.shape[1]
        self_attention = aggregate_attention(controller, res=32, from_where=("up", "down"), is_cross=False)  # [batch, res, res, res**2]
        cross_attention = aggregate_attention(controller, res=16, from_where=("up", "down"), is_cross=True)  # [batch, res, res, n_tokens]
        for i in range(len(self_attention)):
            cluster = self.cluster(self_attention[i]).astype(np.uint8)
            cluster2noun = self.cluster2noun(cluster, cross_attention[i], indices[i])
            cluster2noun = cluster2noun[cluster].astype(np.uint8)
            cluster2noun = np.array(Image.fromarray(cluster2noun).resize((image_res, image_res), resample=2))
            if len(np.unique(cluster2noun)) != 1:
                mask = multi_class_dense_crf(images[i], cluster2noun)
                mask = np.argmax(mask, axis=0)
            else:
                mask = cluster2noun
            label = np.array(labels[i], dtype=np.uint8)
            mask = label[mask]
            clusters.append(cluster)
            out_crfs.append(mask)
        return clusters, out_crfs, self_attention, cross_attention

    def cluster(self, self_attention):
        resolution = self_attention.shape[0]
        attn = self_attention.cpu().numpy().reshape(resolution ** 2, resolution ** 2)
        kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(attn)
        clusters = kmeans.labels_
        clusters = clusters.reshape(resolution, resolution)
        return clusters

    def cluster2noun(self, clusters, cross_attention, indices):
        result = []
        maps = cross_attention.cpu().numpy()[:, :, [i + 1 for i in indices]]
        normalized_maps = np.zeros_like(maps).repeat(2, axis=0).repeat(2, axis=1)
        for i in range(maps.shape[-1]):
            curr_noun_map = maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            normalized_maps[:, :, i] = (curr_noun_map - curr_noun_map.min()) / (curr_noun_map.max() - curr_noun_map.min())
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_maps[:, :, i] for i in range(len(indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result.append(np.argmax(np.array(scores)) + 1 if max(scores) > self.background_segment_threshold else 0)
            
        return np.array(result)

    def get_background_mask(self, obj_token_index):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        mask = clusters.copy()
        obj_segments = [
            c for c in cluster2noun if cluster2noun[c][0] == obj_token_index - 1]
        background_segments = [c for c in cluster2noun if cluster2noun[c]
                               == "BG" or cluster2noun[c][1] in self.background_nouns]
        for c in range(self.num_segments):
            if c in background_segments and c not in obj_segments:
                mask[clusters == c] = 0
            else:
                mask[clusters == c] = 1
        return mask

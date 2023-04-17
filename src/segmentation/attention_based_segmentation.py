from helper.attention_helper import aggregate_attention
import nltk
from sklearn.cluster import KMeans
import numpy as np
import sys


class AttentionBasedSegmentor:
    def __init__(self, method, num_segments, background_segment_threshold, res, background_nouns=[]):
        # fmt: off
        self.method                         = method
        # self.controller                     = controller
        self.num_segments                   = num_segments
        self.background_segment_threshold   = background_segment_threshold
        self.resolution                     = res
        self.background_nouns               = background_nouns
        # self.nouns                          = None 
        # fmt: on

    def __call__(self, prompts, self_attention, cross_attention, token_indices, *args, **kwargs):
        """
        tokenized_prompts: [list of tokens in a given sentence]
        """
        clusters = self.cluster(self_attention)
        cluster2noun = self.cluster2noun(clusters, cross_attention)
        return clusters

    def cluster(self, self_attention):
        """
        Get the semantic segmentation by clustering the self attention
        extract at resolution 32

        Return:
            clusters: np.array: shape [batch, resolution, resolution]
        """
        np.random.seed(1)
        self_attention = self_attention.mean(dim=1)
        B, resolution, _, dim = self_attention.shape
        attn = self_attention.cpu().numpy().reshape(B, resolution ** 2, resolution ** 2)

        outputs = []
        kmeans = KMeans(n_clusters=self.num_segments, n_init=10)

        for attn_map in attn:
            out = kmeans.fit(attn_map)
            out = out.labels_
            out = out.reshape(resolution, resolution)
            outputs.append(out)

        clusters = np.stack(outputs, axis=0)
        return clusters

    def cluster2noun(self, clusters, cross_attention):
        """
        args: 
            clusters: np.array [batch, self_res, self_res]
            cross_attention: torch.Tensor [batch, n_heads, res, res, n_tokens]
        """
        result = []
        nouns_indices = [[index for (index, word) in pair_word_index]
                         for pair_word_index in self.nouns]
        # nouns_indices = [index for (index, word) in self.nouns]

        #   Normalized across the heads dim of the cross attention
        #   Then cross attention will get the shape of [batch, res, res, n_tokens]
        cross_attention = cross_attention.mean(dim=1)
        noun_maps_batch = []
        for idx, cross_attn in enumerate(cross_attention):
            cluster2noun_res = {}
            # print([i + 1 for i in nouns_indices[idx]])
            # print(nouns_indices[idx])
            # noun_maps = cross_attn.cpu().numpy()[:, :, [i + 1 for i in nouns_indices[idx]]]
            # noun_maps_batch.append(noun_maps)

        # noun_maps_batch = np.stack(noun_maps_batch, axis = 0)

            nouns_maps = cross_attn.cpu().numpy(
            )[:, :, [i + 1 for i in nouns_indices[idx]]]
            normalized_nouns_maps = np.zeros_like(
                nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
            for i in range(nouns_maps.shape[-1]):
                curr_noun_map = nouns_maps[:, :, i].repeat(
                    2, axis=0).repeat(2, axis=1)
                normalized_nouns_maps[:, :, i] = (
                    curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
            for c in range(self.num_segments):
                cluster_mask = np.zeros_like(clusters[idx])
                cluster_mask[clusters[idx] == c] = 1
                score_maps = [cluster_mask * normalized_nouns_maps[:, :, i]
                              for i in range(len(nouns_indices[idx]))]
                scores = [score_map.sum() / cluster_mask.sum()
                          for score_map in score_maps]
                cluster2noun_res[c] = self.nouns[np.argmax(np.array(scores))] if max(
                    scores) > self.background_segment_threshold else "BG"
                result.append(cluster2noun_res)
        return result

    def refine_mask_classes(self, clusters, cluster2noun):
        bg_list = [idx for idx, item in enumerate(
            cluster2noun.values()) if isinstance(item, str) and item == 'BG']
        bg_pix = bg_list[0]
        # breakpoint()
        for bg in bg_list:
            try:
                idx_boolean = clusters == bg
                clusters[idx_boolean] = bg_pix
            except Exception as e:
                print(bg)

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

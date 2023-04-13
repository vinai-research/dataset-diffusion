#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def dense_crf(image: np.array, attention: np.array, threshold: float):
    image = np.ascontiguousarray(image)

    labels = (attention > threshold).astype(int)
    
    c = 2
    h = labels.shape[0]
    w = labels.shape[1]

    U = utils.unary_from_labels(labels, 2, 0.7, zero_unsure=False)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def multi_class_dense_crf(image: np.array, attention: np.array, threshold: float):
    if image.dtype == np.float32:
        image = (image * 255.).astype(np.uint8)
    num_classes = attention.shape[0] + 1
    image = np.ascontiguousarray(image)

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
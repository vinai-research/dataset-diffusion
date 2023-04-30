from pathlib import Path


classes=['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor']

ALIAS = {
    'person' : [
        'man',
        'men',
        'women',
        'woman',
        'kids',
        'boy',
        'girl',
        'rider',
        'swimmer',
        'people',
        'children',
    ],

    'aeroplane' : [
        'airplane',
        'plane',
        'aircraft',
        'jet',
    ],

    'diningtable' : [
        'dining table',
        'dining tables',
        'table',
        'desk',
        
    ],

    'chair' : [
        'bench',
        'couch',
    ],

    'motorbike ': [
        'motor',
        'motorcycle',
        'bike',
    ],

    'tvmonitor' : [
        'tv',
        'television',
        'monitor',
        'monitors',
    ],

    'pottedplant' : [
        'plant',
        'plants',
        'potted plant',
        'potted plants',
    ]
}
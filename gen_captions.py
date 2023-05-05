import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import random 
from pathlib import Path
from tqdm import tqdm
import pickle
from nltk import word_tokenize
import pandas as pd
import csv
import json

BASEDIR = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_DIR = BASEDIR / 'data/VOC2012/JPEGImages'
TEXT_DIR = BASEDIR / 'data/VOC2012/NewCaptions1'
TEXT_DIR.mkdir(parents=True, exist_ok=True)
# img_list = [item for item in IMG_DIR.glob('*.jpg')]
# print(len(img_list))

def generate_captions(image_path: Path, save_path: Path):

    processor = BlipProcessor.from_pretrained("/home/ubuntu/DS-Diffusion/pretrained/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("/home/ubuntu/DS-Diffusion/pretrained/blip-image-captioning-base", torch_dtype = torch.float16).to(DEVICE)

    img_files = [item for item in sorted(image_path.glob('*.jpg'))] 

    templates = [
        "a photo of",
        "a picture of",
        "a realistic paint of",
        "a photography of",
        "",
    ]
    result = {}
    for img_file in tqdm(img_files):
        
        list_captions = []
        file_name, _ = str(img_file).split('/')[-1].split('.') 
        raw_image     = Image.open(img_file)
        # conditional image captioning

        out = None
        prompt = ""

        for text in templates:
            if text != "":
                inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16) 
                out = model.generate(**inputs)
                           
            # >>> a photography of a woman and her dog
            else:
            # unconditional image captioning
                inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
                out = model.generate(**inputs) 
                
            prompt = processor.decode(out[0], skip_special_tokens=True)
            list_captions.append(prompt)
        result[file_name] = list_captions 
    breakpoint() 
    with open(save_path / 'voc_captions_new.json', mode = 'w')  as writer:
        json.dump(result, writer)
    writer.close()   
    print('Save result')
    

        # with open(save_path / f'{file_name}.txt', mode = 'w', encoding = 'utf-8') as writer:
        #     writer.write(prompt)
        # writer.close()

def save_to_pkl():
    text_dir = BASEDIR / 'data/VOC2012/NewCaptions1'
    res = {}
    count = 0
    for text_path in text_dir.glob('*.txt'):
        file_name, _ = str(text_path).split('/')[-1].split('.')
        string = None
        with open(text_path, mode = 'r') as reader:
            string = reader.readlines()
            for idx, prompt in enumerate(string):
                string[idx] = prompt.strip('\n')
        reader.close()
        res[file_name] =  string
        count += 1

    assert len(res) == count
    with open("new_caption.pkl", mode = 'wb') as writer:
        pickle.dump(res, writer) 
    writer.close()

classes=('aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor')
people_alias = [
    'man',
    'men'
    'women',
    'woman'
    'kids',
    'boy',
    'girl',
    'rider',
    'swimmer',
    'people',
    'children',
]

plane_alias = [
    'airplane',
    'plane',
    'aircraft',
    'jet',
]

table_alias = [
    'table',
    'desk',
]

chair_alias = [
    'bench',
    'couch',
]

motor_alias = [
    'motor',
    'motorcycle',
    'bike',
]
def stats_caption(path: Path):
    out = {}
    string = None
    for item in path.glob('*.txt'):
        with open(item, mode = 'r', encoding='utf-8') as reader:
            string = reader.readlines()
        reader.close()
        for item in string:
            item = item.strip('\n')
            for token in word_tokenize(item):
                if token in classes:
                    if token not in out:
                        out[token] = 0
                    else:
                        out[token] += 1

                elif token in people_alias:
                    if 'person' not in out:
                        out['person'] = 0
                    else:
                        out['person'] += 1
                
                # elif token in plane_alias:
                #     if 'aeroplane' not in out:
                #         out['aeroplane'] = 0
                #     else:
                #         out['aeroplane'] += 1

                # elif token in table_alias:
                #     if 'diningtable' not in out:
                #         out['diningtable'] = 0
                #     else:
                #         out['diningtable'] += 1

                # elif token in chair_alias:
                #     if 'chair' not in out:
                #         out['chair'] = 0
                #     else:
                #         out['chair'] += 1
                # elif token in motor_alias:
                #     if 'motorbike' not in out:
                #         out['motorbike'] = 0
                #     else:
                #         out['motorbike'] += 1
                # elif token in chair_alias:
                #     if 'sofa' not in out:
                #         out['sofa'] = 0
                #     else:
                #         out['sofa'] += 1
    breakpoint()
    return out

def stats_caption_json(json_file):
    import json
    Dict = None
    out = {}

    with open(json_file, mode = 'r', encoding='utf-8') as reader:
        Dict = json.load(reader)
    for _dict in Dict:
        item = _dict['caption']

        item = item.strip('\n')
        for token in word_tokenize(item):
            if token in classes:
                if token not in out:
                    out[token] = 0
                else:
                    out[token] += 1

            elif token in people_alias:
                if 'person' not in out:
                    out['person'] = 0
                else:
                    out['person'] += 1
            
            elif token in plane_alias:
                if 'aeroplane' not in out:
                    out['aeroplane'] = 0
                else:
                    out['aeroplane'] += 1

            elif token in table_alias:
                if 'diningtable' not in out:
                    out['diningtable'] = 0
                else:
                    out['diningtable'] += 1

            elif token in chair_alias:
                if 'chair' not in out:
                    out['chair'] = 0
                else:
                    out['chair'] += 1
            elif token in motor_alias:
                if 'motorbike' not in out:
                    out['motorbike'] = 0
                else:
                    out['motorbike'] += 1
            elif token in chair_alias:
                if 'sofa' not in out:
                    out['sofa'] = 0
                else:
                    out['sofa'] += 1
    out = [out]
    with open('test.csv', mode = 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=classes)
        writer.writeheader()
        writer.writerows(out)
    print(out)
    breakpoint()

def convert_text_prompt(json_file):
    obj = None
    with open(json_file, mode = 'r') as reader:
        obj = json.load(reader)
    reader.close()
    
    prompts = []
    for file_name, list_sent in obj.items():
        

        for idx, sent in enumerate(list_sent):
            res = {}
            name = f'{file_name}_{idx}.jpg'
            res["filename"] = name
            res["caption"] = sent

            prompts.append(res)
    
    with open(f'{BASEDIR}/data/552023_VOC_captions.json', mode = 'w') as writer:
        json.dump(prompts, writer)
    writer.close()
    print(type(obj))
    breakpoint()
if __name__ == '__main__' :

    path = BASEDIR / 'data/voc_captions_new.json'
    convert_text_prompt(path)
    # stats_caption_json(BASEDIR / 'data/voc_captions.json')
    # stats = stats_caption(TEXT_DIR)
    # breakpoint()
    # print(stats)
    # genedf.torate_captions(IMG_DIR, TEXT_DIR)
    # save_to_pkl()
    # import json
    # generate_captions(IMG_DIR, BASEDIR / 'data')
    # prompt = None
    # with open('new_caption.pkl', mode = 'rb') as reader:
    #     prompt = pickle.load(reader)
    #     # b = json.dumps(prompt)
    # reader.close()

    # with open('new_caption.json', mode = 'w') as json_file:
    #     json.dump(prompt, json_file)

    # breakpoint() 

    
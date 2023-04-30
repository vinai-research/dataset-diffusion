from urllib.request import urlretrieve
import pandas as pd
from const import *
import json 
from pathlib import Path
from nltk.tokenize import word_tokenize

BASE = Path(__file__).resolve().parent.parent
RANGE = 1000

def open_json(path):
    obj = None
    with open(path, mode = 'r') as reader:
        obj = json.load(reader) 
    reader.close()
    return obj

def write_json(path, json_obj):
    try:
        with open(path, mode = 'w') as writer:
            json.dump(json_obj, writer)
        writer.close()
        return True
    except Exception:
        return False

def get_prompts_diffusiondb(metadata: Path):
    path = str(metadata)
    df = pd.read_parquet(path)
    json_file = df['prompt'][:RANGE].to_list()       #   List of prompt retrieve from the diffusion db
    
    #   Init the mapping dict of prompt
    retrieve_prompts = {}
    for item in classes:
        retrieve_prompts[item] = []

    for prompt in json_file:
        tokens = word_tokenize(prompt)

        for class_name, alias in ALIAS.items():
            if class_name in tokens or any(x in tokens for x in alias):
                retrieve_prompts[class_name].append(prompt)

    return retrieve_prompts 

if __name__ == '__main__':

    metadata_path = BASE / 'metadata.parquet' 
    prompts = get_prompts_diffusiondb(metadata_path)

    path = BASE / 'data/diffusion_db.json'
    write_json(path, prompts)
    breakpoint()
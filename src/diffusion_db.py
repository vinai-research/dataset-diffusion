from urllib.request import urlretrieve
import pandas as pd

# Download the parquet table
# table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
# urlretrieve(table_url, 'metadata.parquet')
classes=['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor']

RANGE = 100000
# # Read the table using Pandas
metadata_df = pd.read_parquet('metadata.parquet')
json_file = metadata_df['prompt'][:RANGE].to_dict()
# breakpoint()
path = 'prompt_diffusiondb.json'
import json
obj = None
with open(path, 'w') as writer:
    json.dump(json_file, writer)
writer.close()
with open(path, mode = 'r') as reader:
    obj = json.load(reader)
reader.close()

for key, value in obj.items():
    print(value)
breakpoint()
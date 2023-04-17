from torch.utils.data import Dataset
import torch
import torchvision
from pathlib import Path
from typing import List


class CaptionDataset(Dataset):
    def __init__(self, root: Path):
        self.root: Path = root
        self.file_list = self.get_list_files()

    def get_list_files(self) -> List[Path]:

        out = [item for item in self.root.glob('*.txt')]
        assert len(out) > 0
        return out

    def load_file(self, path) -> List[str]:
        txt = None
        try:
            with open(path, mode='r', encoding='utf-8') as reader:
                txt = reader.readlines()[0].strip('\n')
            reader.close()
        except Exception as e:
            raise BufferError(e)

        return txt

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):

        file_name = str(self.file_list[idx]).split('/')[-1].split('.')[0]
        txt = self.load_file(self.file_list[idx])

        return file_name, txt


class DictCaptionDataset(Dataset):
    def __init__(self, captions: List[dict]):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx: int):
        return self.captions[idx]['filename'], self.captions[idx]['caption']

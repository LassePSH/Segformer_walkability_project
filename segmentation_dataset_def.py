import torch 
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

#costum dataset class
class Dataset():
    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        file_id = self.img_ids[item]
        file_id = str(int(file_id))

        return {
         'img_id': file_id   
        }

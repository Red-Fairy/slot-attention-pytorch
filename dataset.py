import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class CLEVR(Dataset):
    def __init__(self, root, split='train'):
        super(CLEVR, self,).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = root  
        self.files = os.listdir(os.path.join(self.root_dir, self.split, 'images'))
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, self.split, 'images', path)).convert('RGB')
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)

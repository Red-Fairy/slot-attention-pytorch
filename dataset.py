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
        self.transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, self.split, 'images', path)).convert('RGB')
        image = image.resize((128 , 128))
        image = self.transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    def filter_out(x):
        if not x.endswith('png'):
            return False
        filter_names = ['albeda', 'depth', 'flat', 'normal', 'shadow']
        for name in filter_names:
            if name in x:
                return False
        return True

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if filter_out(fname):
                path = os.path.join(root, fname)
                images.append(path)
    print(len(images))
    return images

class CLEVRTex(Dataset):
    def __init__(self, root):
        super(CLEVRTex, self,).__init__()
        
        self.root_dir = root
        self.files = make_dataset(root)
        self.transform = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        # image = image.resize((192, 256))
        image = self.transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)

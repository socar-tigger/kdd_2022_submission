import os 
import random 
import pandas as pd 
from pathlib import Path
from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

name_to_class = {
    'outer_normal': 0,
    'outer_damage': 1,
    'outer_dirt': 2,
    'outer_wash': 3, 
    'inner_wash':4, 
    'inner_dashboard':5,
    'inner_cupholder':6,
    'inner_cupholder_dirt':7,
    'inner_glovebox':8,
    'inner_washer_fluid':9,
    'inner_front_seat':10,
    'inner_rear_seat':11,
    'inner_trunk':12,
    'inner_sheet_dirt':13,
}

class SofarDataset(Dataset):
    def __init__(self, root, dataset, mode, train_class, test_class=None, transform=None, ce_label=False):
        self.root = root
        self.data = pd.read_csv(os.path.join(root, 'sofar_dataset', '{}_data_{}.csv'.format(mode, dataset.split('_')[-1])), header=0)
        
        self.transform = transform
        
        # select only target classes
        train_class.sort()
        if mode == 'train':
            self.data = self.data[self.data.label.isin(train_class)]
        else:
            self.data = self.data[self.data.label.isin(test_class)]

        # cross entropy 학습을 위한 label 변화(0~num_class)
        if ce_label:
            assert test_class == None or len(train_class) >= len(test_class)
            for new_c, c in enumerate(train_class):
                self.data['label'] = self.data['label'].apply(lambda x: new_c if x == c else x)
        
    def __getitem__(self, index):
        filename, label = self.data.iloc[index]
        path = os.path.join(self.root, 'sofar_dataset', filename)
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    def __len__(self):
        return len(self.data)


def create_dataloader(args):
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    
    
    train_dataset = SofarDataset(args.data_root_path, 
                                    args.dataset,
                                    mode='train', 
                                    train_class = args.train_class, 
                                    transform=train_transform,
                                    ce_label=args.ce_label)
    test_dataset = SofarDataset(args.data_root_path, 
                                    args.dataset,
                                    mode='test', 
                                    train_class = args.train_class,
                                    test_class = args.test_class,
                                    transform=test_transform,
                                    ce_label=args.ce_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )        
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )        

    return train_loader, test_loader

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                           y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size,
                y * block_size:(y + 1) * block_size] = temp

    return jigsaws
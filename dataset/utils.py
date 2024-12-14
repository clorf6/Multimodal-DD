import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from .coco import coco_train, coco_eval
from .sync import sync_set
from .augment import RandomAugment

def collate_fn_train(batch):
    images = torch.stack([item[0] for item in batch]) 
    captions = [caption for item in batch for caption in item[1]]
    return images, captions

def collate_fn_syn(batch):
    images = torch.stack([item[0] for item in batch]) 
    captions = [item[1] for item in batch]
    return images, captions

def load_trainloader(args):  
    if args.dataset == 'coco':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_train = transforms.Compose([           
            transforms.Resize((args.image_size, args.image_size)),          
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ]) 
        train_dataset = coco_train(transform_train, args.dataset_root)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn_train, pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    if args.dataset == 'sync':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])  
        train_dataset = sync_set(transform_test, args.dataset_root)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn_syn, pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    return train_loader

def load_testloader(args):  
    if args.dataset == 'coco':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])  
        test_dataset = coco_eval(transform_test, args.dataset_root, 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)        
    else:
        raise RuntimeError
    return test_loader
import os
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from .coco import coco_train
from .augment import RandomAugment

def load_dataset(args, min_scale=0.5):  
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root=args.image_root, train=True, download=False,
                                    transform=transform_train)
    if args.dataset == 'coco':
        transform_train = transforms.Compose([           
            transforms.Resize((args.image_size, args.image_size)),          
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])   
        train_dataset = coco_train(transform_train, os.path.join(args.image_root, 'train'), args.ann_root)
    
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    return loader


def gen_label_list(args):
    df = pd.read_csv(args.label_file_path)

    labels = dict()
    for _, row in df.iterrows():
        labels[row['id']] = row['label']

    return labels

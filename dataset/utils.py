import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from .coco import coco_train
from .augment import RandomAugment

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # Stack image tensors
    labels = torch.tensor([item[1] for item in batch])  # Stack labels into a tensor
    captions = [caption for item in batch for caption in item[2]]
    return images, labels, captions

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
        train_dataset = coco_train(transform_train, os.path.join(args.image_root, 'train'), 
                                   args.ann_root, args.num_label_clusters)
    
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    return loader, train_dataset.center

def gen_label_list(label_file_path):
    df = pd.read_csv(label_file_path)

    labels = dict()
    for _, row in df.iterrows():
        labels[row['id']] = row['label']

    return labels

def gen_label_name(label_name_path):
    df = pd.read_csv(label_name_path)

    labels = dict()
    for _, row in df.iterrows():
        labels[row['id']] = row['name']

    return labels

def save_label_names(label_file_path, save_file_path, label_centers):
    label_names = {}
    label_dict = gen_label_list(label_file_path)
    for i in range(len(label_centers)):
        label_centers[i] = label_centers[i].tolist()
        threshold = label_centers[i].max() * 0.5
        label_indices = [x + 1 for x, num in enumerate(label_centers[i]) if num >= threshold]
        label_names[i] = ','.join([label_dict[index] for index in label_indices])
    os.makedirs(save_file_path, exist_ok=True)
    save_path = os.path.join(save_file_path, 'label_names.csv')
    pd.DataFrame(label_names.items(), columns=['id', 'name']).to_csv(save_path, index=False)
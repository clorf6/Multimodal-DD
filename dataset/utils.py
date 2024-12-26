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
from transformers import CLIPProcessor, CLIPModel

def collate_fn_train(batch):
    images = torch.stack([item[0] for item in batch]) 
    captions = [item[1] for item in batch]
    indices = [item[2] for item in batch]
    return images, captions, indices

def collate_fn_syn(batch):
    images = torch.stack([item[0] for item in batch]) 
    captions = [item[1] for item in batch]
    return images, captions

def load_train_dataset(args):
    transform = []
    if args.is_resize:
        transform.append(transforms.Resize((args.image_size, args.image_size)))
    if args.is_augment:
        transform.append(transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']))
    transform.append(transforms.ToTensor())
    if args.is_normalize:
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform.append(normalize)
    transform_train = transforms.Compose(transform) 
    if args.dataset == 'coco':
        train_dataset = coco_train(transform_train, args.dataset_root)
    elif args.dataset == 'sync':
        train_dataset = sync_set(transform_train, args.dataset_root)
    else:
        raise RuntimeError
    return train_dataset

def load_train_loader(args):  
    train_dataset = load_train_dataset(args)
    if args.dataset == 'coco':
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn_train, pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    elif args.dataset == 'sync':
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn_syn, pin_memory=True, shuffle=args.shuffle, drop_last=args.drop_last)        
    else:
        raise RuntimeError
    return train_loader

def load_test_loader(args):  
    if args.dataset == 'coco':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ])  
        test_dataset = coco_eval(transform_test, args.dataset_root, 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, 
                                 shuffle=False, drop_last=False)        
    else:
        raise RuntimeError
    return test_loader

def load_clip_model(device, args):
    clip_model = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    return clip_model, clip_processor

def compute_clip_score(clip_model, clip_processor, image, text, device):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    similarity = torch.cosine_similarity(image_embeds, text_embeds, dim=-1)
    return similarity.mean().cpu().item()
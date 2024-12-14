import json
import os
from PIL import Image
from torch.utils.data import Dataset
import re

def pre_caption(caption,max_words=30):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class coco_train(Dataset):
    def __init__(self, transform, dataset_root, max_words=30, prompt=''):
        self.data = json.load(open(os.path.join(dataset_root, 'coco_karpathy_train.json'), 'r'))
        self.transform = transform
        self.image_root = dataset_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.data[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        captions = self.data[index]['captions']
        for caption in captions:
            caption = self.prompt + pre_caption(caption, self.max_words)
        if len(captions) < 5:
            captions += [] * (5 - len(captions))
        else:
            captions = captions[:5]
        assert len(captions) == 5
        return image, captions
            
    
class coco_eval(Dataset):
    def __init__(self, transform, dataset_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        split (string): val or test
        '''
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        self.annotation = json.load(open(os.path.join(dataset_root, filenames[split]),'r'))
        self.transform = transform
        self.image_root = dataset_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for _, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
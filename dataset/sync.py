from torch.utils.data import Dataset
from PIL import Image
import json
import re
import os

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

class sync_set(Dataset):
    def __init__(self, transform, dataset_root, max_words=30):  
        
        self.annotation = json.load(open(os.path.join(dataset_root, 'results.json'), 'r'))
        self.transform = transform
        self.img_root = os.path.join(dataset_root, "images")
        self.max_words = max_words
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.img_root, self.annotation[index]['image_path'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, pre_caption(self.annotation[index]['caption'][0], self.max_words)
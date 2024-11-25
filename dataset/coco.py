import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class coco_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.caption = json.load(open(os.path.join(ann_root, 'train_data.json'), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.caption['annotations'])

    def __getitem__(self, index):
        image_id = int(self.caption['annotations'][index]['image_id'])
        image_path = os.path.join(self.image_root, f"COCO_train2014_{image_id:0>12}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = np.array(self.caption['annotations'][index]['label'], dtype=int) - 1
        onehot = np.zeros(90, dtype=bool)
        onehot[label] = True
        onehot = ''.join('1' if x else '0' for x in onehot)
        caption = ''.join(self.caption['annotations'][index]["captions"])
        # TODO: embedding caption

        return image, onehot, caption
    
    @property
    def label(self):
        index = 0
        while index < len(self.caption['annotations']):
            label = np.array(self.caption['annotations'][index]['label'], dtype=int) - 1
            onehot = np.zeros(90, dtype=bool)
            onehot[label] = True
            onehot = ''.join('1' if x else '0' for x in onehot)
            yield onehot
            index += 1

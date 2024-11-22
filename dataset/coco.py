import json
import os

from PIL import Image
from torch.utils.data import Dataset


class coco_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.annotation = json.load(open(os.path.join(ann_root, 'instances_train.json'), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation['annotations'])

    def __getitem__(self, index):
        image_id = int(self.annotation['annotations'][index]['image_id'])
        image_path = os.path.join(self.image_root, f"COCO_train2014_{image_id:0>12}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # TODO: embedding caption

        return image, self.annotation['annotations'][index]['category_id']

import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from tqdm import tqdm

class coco_train(Dataset):
    def __init__(self, transform, image_root, ann_root, num_label_clusters, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.caption = json.load(open(os.path.join(ann_root, 'train_data.json'), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        self.model = MiniBatchKMeans(n_clusters=num_label_clusters, random_state=0, 
                                     batch_size=(num_label_clusters * 10), n_init="auto")
        tmp = []
        for onehot in tqdm(self.onehots, total=len(self.caption['annotations'])):
            tmp.append(onehot)
            if len(tmp) == num_label_clusters * 10:
                self.model.partial_fit(np.vstack(tmp))
                tmp = []
        self.model.partial_fit(np.vstack(tmp))
        
        self.center = self.model.cluster_centers_
        
    def to_onehot(self, label):
        onehot = np.zeros(90, dtype=bool)
        onehot[label - 1] = True
        return onehot

    def __len__(self):
        return len(self.caption['annotations'])

    def __getitem__(self, index):
        image_id = int(self.caption['annotations'][index]['image_id'])
        image_path = os.path.join(self.image_root, f"COCO_train2014_{image_id:0>12}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        onehot = self.to_onehot(np.array(self.caption['annotations'][index]['label'], dtype=int))
        label = self.model.predict(onehot.reshape(1, -1))[0]
        caption = self.caption['annotations'][index]["captions"]
        if len(caption) < 5:
            caption += [] * (5 - len(caption))
        else:
            caption = caption[:5]
        assert len(caption) == 5
        return image, label, caption

    @property
    def onehots(self):
        index = 0
        while index < len(self.caption['annotations']):
            yield self.to_onehot(np.array(self.caption['annotations'][index]['label'], dtype=int))
            index += 1
            
    
'''
Generate prototype using the diffusers pipeline
Author: Su Duo & Houjunjie
Date: 2023.9.21
'''

import numpy as np
import torch
import os
import pickle
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from scripts.pipeline_stable_diffusion_gen_latents import StableDiffusionGenLatentsPipeline
import argparse
import json
import time
import warnings
from sklearn.cluster import KMeans
from sklearn_extra.cluster import CLARA
from tqdm import tqdm

from dataset.utils import load_train_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/nas-new/home/yangnianzu/jsl/Multimodal-DD', type=str, 
                        help='root path')
    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='train batch size')
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5', type=str,
                        help='model name')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to distillate')
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size')
    parser.add_argument('--result_num', default=500, type=int,
                        help='number of results')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='number of workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle the data')
    parser.add_argument('--time_str', default=None, type=str,
                        help='time str')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='drop the last batch')
    parser.add_argument('--strength', '-s', default=0.7, type=float, 
                        help='diffusers strength')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument("--delay", default=0, type=int,
                        help="Delay in seconds before starting computation")
    parser.add_argument("--is_resize", default=True, type=bool,
                        help="resize image")
    parser.add_argument("--is_augment", default=True, type=bool,
                        help="use data augmentation")
    parser.add_argument("--is_normalize", default=True, type=bool,
                        help="use data augmentation")
    args = parser.parse_args()
    if args.delay > 0:
        time.sleep(args.delay)
    if args.time_str is None:
        timestr = time.ctime().replace(f' ', f'-').replace(f':', f'-')
    else:
        timestr = args.time_str
    print("Time: ", timestr)
    model = args.model.split('/')[-1]
    args.dataset_root = os.path.join(args.root, 'data', args.dataset)
    args.save_path = os.path.join(args.root, 'results', args.dataset, timestr)
    args.cache_path = os.path.join(args.root, 'cache', f"{model}-strength{args.strength}-guidance{args.guidance_scale}")
    args.model_path = os.path.join(args.root, 'models', model)
    return args


def initialize_km_models(args):
    model = CLARA(n_clusters=args.result_num, random_state=0)
    return model


def prototype_kmeans(pipe, data_loader, km_model, args):
    latent_images = []
    latent_indices = []
    image_cache_path = os.path.join(args.cache_path, 'image_cache.pkl')
    index_cache_path = os.path.join(args.cache_path, 'index_cache.pkl')
    if os.path.isfile(image_cache_path) and os.path.isfile(index_cache_path):
        with open(image_cache_path, 'rb') as f:
            latent_images = pickle.load(f)
        with open(index_cache_path, 'rb') as f:
            latent_indices = pickle.load(f)
        print("cache loaded")
    else: 
        os.makedirs(args.cache_path, exist_ok=True)
        for batch_images, _, batch_indices in tqdm(data_loader, total=len(data_loader), position=0):
            B = batch_images.size(0)
            batch_images = batch_images.cuda(non_blocking=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                image_embed = pipe(image=batch_images, strength=args.strength, type='image').view(B, -1)
            for i in range(B):
                latent_images.append(image_embed[i].view(1, -1).cpu().numpy())
                latent_indices.append(batch_indices[i])
        with open(image_cache_path, 'wb') as f:
            pickle.dump(latent_images, f)
        with open(index_cache_path, 'wb') as f:
            pickle.dump(latent_indices, f)

    km_model.fit(np.vstack(latent_images))
    return km_model, latent_images, latent_indices

def gen_prototype(km_models, latent_images, latent_indices):
    prototype = {}
    model = km_models
    N = 64
    if hasattr(model, 'medoid_indices_'):
        medoids_idxs = model.medoid_indices_.tolist()
        image_centers = []
        index_centers = []
        for idx in medoids_idxs:
            image_center = latent_images[idx].reshape(4, N, N)
            image_centers.append(image_center.tolist())
            index_centers.append(latent_indices[idx])
        prototype = (image_centers, index_centers)
    else:
        print(f"Warning: model has no attribute cluster_centers_")
    return prototype

def save_prototype(prototype, args):
    save_prototype_path = os.path.join(args.save_path, 'prototypes')
    os.makedirs(save_prototype_path, exist_ok=True)
    json_file = os.path.join(save_prototype_path, f'{args.dataset}-res{args.result_num}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {save_prototype_path}")

def main():
    # 1.parse arguments
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Device: ", args.device)

    # 2.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16
    )
    pipe = pipe.to(args.device)

    # 3.obtain training data
    trainloader = load_train_loader(args)

    # 4.initialize & run partial k-means model each class
    km_model = initialize_km_models(args)
    fitted_km, latent_images, latent_indices = prototype_kmeans(pipe=pipe, data_loader=trainloader, km_model=km_model, args=args)

    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(fitted_km, latent_images, latent_indices)
    save_prototype(prototype, args)

if __name__ == "__main__" :
    main()

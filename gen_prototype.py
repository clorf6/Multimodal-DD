'''
Generate prototype using the diffusers pipeline
Author: Su Duo & Houjunjie
Date: 2023.9.21
'''

import numpy as np
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_gen_latents import StableDiffusionGenLatentsPipeline
import argparse
import json
import time
import warnings
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset.utils import load_dataset, save_label_names

def parse_args():
    timestr = time.ctime().replace(f' ', f'-').replace(f':', f'-')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to distillate')
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size')
    parser.add_argument('--image_root', default='/home/xun_ying/DD/data/coco', type=str,
                        help='image root dir')
    parser.add_argument('--ann_root', default='/home/xun_ying/DD/data/coco/annotations', type=str,
                        help='annotation root dir')
    parser.add_argument('--ipc', default=1, type=int,
                        help='image per class')
    parser.add_argument('--km_expand', default=10, type=int,
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--num_label_clusters', default=500, type=int,
                        help='number of label clusters')
    parser.add_argument('--label_file_path', default='/home/xun_ying/DD/data/coco/coco.csv', type=str,
                        help='root dir')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle the data')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='drop the last batch')
    parser.add_argument('--save_prototype_path', default=os.path.join('/home/xun_ying/DD/data/coco/prototypes', timestr), type=str,
                        help='where to save the generated prototype json files')
    args = parser.parse_args()
    return args


def initialize_km_models(args):
    km_models = {}
    for label in range(args.num_label_clusters):
        model_name = f"MiniBatchKMeans_{label}"
        model = MiniBatchKMeans(n_clusters=args.ipc, random_state=0, batch_size=(
            args.km_expand * args.ipc), n_init="auto")
        km_models[model_name] = model
    return km_models


def prototype_kmeans(pipe, data_loader, km_models, args):
    latents = []
    for label in range(args.num_label_clusters):
        latents.append([])
    for images, labels, captions in tqdm(data_loader, total=len(data_loader), position=0):
        B = images.size(0)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            image_embed = pipe(prompt=captions[0], image=images, strength=0.7, type='image')
            prompt_embed = pipe(prompt=captions, strength=0.7, guidance_scale=8, type='text')
            prompt_embed = prompt_embed.view(B, 5, 2, -1)
            prompt_embed = prompt_embed.mean(dim=1)
            latent_embed = torch.cat([image_embed.view(B, -1), prompt_embed.view(B, -1)], dim=-1)
        for latent, label in zip(latent_embed, labels):
            latent = latent.view(1, -1).cpu().numpy()
            latents[label].append(latent)
            if len(latents[label]) == args.km_expand * args.ipc:
                km_models[f"MiniBatchKMeans_{label}"].partial_fit(np.vstack(latents[label]))
                latents[label] = []  # save the memory, avoid repeated computation
    
    for label in range(args.num_label_clusters):
        if len(latents[label]) >= args.ipc:
            km_models[f"MiniBatchKMeans_{label}"].partial_fit(np.vstack(latents[label]))
    
    return km_models


def gen_prototype(num_labels, km_models):
    prototype = {}
    for label in range(num_labels):
        model_name = f"MiniBatchKMeans_{label}"
        model = km_models[model_name]
        N = 64
        cluster_centers = model.cluster_centers_
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i][:4 * N * N].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[label] = reshaped_centers
    return prototype


def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-kmexpand{args.km_expand}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.obtain training data
    trainloader, label_centers = load_dataset(args)

    save_label_names(args.label_file_path, args.save_prototype_path, label_centers)
    # 3.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True, torch_dtype=torch.float16
    )
    pipe = pipe.to(args.device)

    # 4.initialize & run partial k-means model each class
    km_models = initialize_km_models(args)
    fitted_km = prototype_kmeans(pipe=pipe, data_loader=trainloader, km_models=km_models, args=args)

    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(args.num_label_clusters, fitted_km)
    save_prototype(prototype, args)

if __name__ == "__main__" :
    main()

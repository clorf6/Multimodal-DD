'''
Generate prototype using the diffusers pipeline
Author: Su Duo & Houjunjie
Date: 2023.9.21
'''

import numpy as np
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from scripts.pipeline_stable_diffusion_gen_latents import StableDiffusionGenLatentsPipeline
import argparse
import json
import time
import warnings
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset.utils import load_trainloader

def parse_args():
    timestr = time.ctime().replace(f' ', f'-').replace(f':', f'-')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', default=16, type=int,
                        help='train batch size')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to distillate')
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size')
    parser.add_argument('--result_num', default=200, type=int,
                        help='number of results')
    parser.add_argument('--km_expand', default=15, type=int,
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle the data')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='drop the last batch')
    args = parser.parse_args()
    args.image_root = os.path.join('/home/xun_ying/DD/data', args.dataset)
    args.save_prototype_path = os.path.join('/home/xun_ying/DD/results', args.dataset, timestr, 'prototypes')
    return args


def initialize_km_models(args):
    model = MiniBatchKMeans(n_clusters=args.result_num, random_state=0, batch_size=(
            args.km_expand * args.result_num), n_init="auto")
    return model


def prototype_kmeans(pipe, data_loader, km_model, args):
    latents = []
    for images, captions in tqdm(data_loader, total=len(data_loader), position=0):
        B = images.size(0)
        negative_prompt = ['cartoon, anime, painting'] * (5 * B)
        images = images.cuda(non_blocking=True)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            image_embed = pipe(image=images, strength=0.7, type='image')
            prompt_embed = pipe(prompt=captions, strength=0.7, guidance_scale=8, negative_prompt=negative_prompt, type='text')
            prompt_embed = prompt_embed.view(B, 5, 2, -1)
            prompt_embed = prompt_embed.mean(dim=1)
            latent_embed = torch.cat([image_embed.view(B, -1), prompt_embed.view(B, -1)], dim=-1)
        for latent in latent_embed:
            latent = latent.view(1, -1).cpu().numpy()
            latents.append(latent)
            if len(latents) == args.km_expand * args.result_num:
                km_model.partial_fit(np.vstack(latents))
                latents = []  # save the memory, avoid repeated computation
    
    if len(latents) >= args.result_num:
        km_model.partial_fit(np.vstack(latents))

    return km_model


def gen_prototype(km_models):
    prototype = {}
    model = km_models
    N = 64
    if hasattr(model, 'cluster_centers_'):
        cluster_centers = model.cluster_centers_
        num_clusters = cluster_centers.shape[0]
        image_centers = []
        caption_centers = []
        for i in range(num_clusters):
            image_center = cluster_centers[i][:4 * N * N].reshape(4, N, N)
            image_centers.append(image_center.tolist())
            caption_center = cluster_centers[i][4 * N * N:].reshape(2, -1, 768)
            caption_centers.append(caption_center.tolist())
        prototype = (image_centers, caption_centers)
    else:
        print(f"Warning: model has no attribute cluster_centers_")
    return prototype


def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-res{args.result_num}-kmexpand{args.km_expand}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")


def main():
    # 1.parse arguments
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.obtain training data
    trainloader = load_trainloader(args)

    # 3.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True, torch_dtype=torch.float16
    )
    pipe = pipe.to(args.device)

    # 4.initialize & run partial k-means model each class
    km_model = initialize_km_models(args)
    fitted_km = prototype_kmeans(pipe=pipe, data_loader=trainloader, km_model=km_model, args=args)

    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(fitted_km)
    save_prototype(prototype, args)

if __name__ == "__main__" :
    main()

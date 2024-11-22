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
import math
import time
import warnings
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset.utils import load_dataset, gen_label_list


def parse_args():
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
    parser.add_argument('--ipc', default=10, type=int,
                        help='image per class')
    parser.add_argument('--km_expand', default=1, type=int,
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='/home/xun_ying/DD/data/coco/coco.csv', type=str,
                        help='root dir')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle the data')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='drop the last batch')
    parser.add_argument('--save_prototype_path', default='/home/xun_ying/DD/data/coco/prototypes', type=str,
                        help='where to save the generated prototype json files')
    args = parser.parse_args()
    return args


def initialize_km_models(label_list, args):
    km_models = {}
    for label in label_list.values():
        model_name = f"MiniBatchKMeans_{label}"
        model = MiniBatchKMeans(n_clusters=args.ipc, random_state=0, batch_size=(
            args.km_expand * args.ipc), n_init="auto")
        km_models[model_name] = model
    return km_models


def prototype_kmeans(pipe, data_loader, label_list, km_models, args):
    latents = {}
    for label in label_list.values():
        latents[label] = []

    for images, label_ids in tqdm(data_loader, total=len(data_loader), position=0):
        images = images.cuda(non_blocking=True)
        label_ids = label_ids.cuda(non_blocking=True)
        labels = []
        for label_id in label_ids:
            label = label_list[label_id.item()]
            labels.append(label)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            init_latents, _ = pipe(prompt=labels, image=images, strength=0.7, guidance_scale=8)

        for latent, prompt in zip(init_latents, labels):
            latent = latent.view(1, -1).cpu().numpy()
            latents[prompt].append(latent)
            if len(latents[prompt]) == args.km_expand * args.ipc:
                km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
                latents[prompt] = []  # save the memory, avoid repeated computation
    if len(latents[prompt]) >= args.ipc:
        km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
    return km_models


def gen_prototype(label_list, km_models):
    prototype = {}
    for label in label_list.values():
        model_name = f"MiniBatchKMeans_{label}"
        model = km_models[model_name]
        cluster_centers = model.cluster_centers_
        N = int(math.sqrt(cluster_centers.shape[1] / 4))
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[label] = reshaped_centers
    return prototype


def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    timestr = time.ctime().replace(f' ', f'-').replace(f':', f'-')
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-kmexpand{args.km_expand}-{timestr}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.obtain label-prompt list
    label_list = gen_label_list(args) # TODO: modify the path in args

    # 2.obtain training data
    trainloader = load_dataset(args)

    # 3.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True, torch_dtype=torch.float16
    )
    pipe = pipe.to(args.device)

    # 4.initialize & run partial k-means model each class
    km_models = initialize_km_models(label_list, args)
    fitted_km = prototype_kmeans(pipe=pipe, data_loader=trainloader, label_list=label_list, km_models=km_models, args=args)

    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(label_list, fitted_km)
    save_prototype(prototype, args)


if __name__ == "__main__" :
    main()

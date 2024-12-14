from scripts.pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline

import torch

import argparse
from dataset.utils import *
import json
import os
import numpy as np
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', type=str, 
                        help='data prepare to distillate')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument('--time_str', default='Fri-Dec-13-15-24-45-2024', type=str,
                        help='time str')
    parser.add_argument('--config_name', default='coco-res200-kmexpand15', type=str,
                        help='configs name')
    parser.add_argument('--strength', '-s', default=0.7, type=float, 
                        help='diffusers strength')
    args = parser.parse_args()
    args.prototype_path = f"/home/xun_ying/DD/results/{args.dataset}/{args.time_str}/prototypes/{args.config_name}.json"
    args.image_path = f"/home/xun_ying/DD/results/{args.dataset}/{args.time_str}/images"
    os.makedirs(args.image_path, exist_ok=True)
    return args


def load_prototype(args):
    prototype_file_path = args.prototype_path
    with open(prototype_file_path, 'r') as f:
        prototype = tuple(json.load(f))
    prototypes = []
    for (image, caption) in zip(prototype[0], prototype[1]):
        image = torch.tensor(image, dtype=torch.float16).to(args.device)
        caption = torch.tensor(caption, dtype=torch.float16).to(args.device)
        prototypes.append((image, caption))
    print("prototype loaded.")
    return prototypes


def gen_syn_images(pipe, prototypes, args):
    for i, (image_embed, caption_embed) in tqdm(enumerate(prototypes), total=len(prototypes), position=0):
        image = image_embed.unsqueeze(0)
        negative = caption_embed[0].unsqueeze(0)
        caption = caption_embed[1].unsqueeze(0)
        images = pipe(latents=image, prompt_embeds=caption, negative_prompt_embeds=negative, is_init=True, strength=args.strength, guidance_scale=args.guidance_scale).images
        images[0].resize((224, 224)).save(os.path.join(args.image_path, f"{i}.png"))


def main():
    # 1.parse args
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.define the diffusers pipeline
    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
    )
    pipe = pipe.to(args.device)

    # 3.load prototypes from json file
    prototypes = load_prototype(args)

    # 4.generate initialized synthetic images and save them for refine
    gen_syn_images(pipe=pipe, prototypes=prototypes, args=args)


if __name__ == "__main__" : 
    main()
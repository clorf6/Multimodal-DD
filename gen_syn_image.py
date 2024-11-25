from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline

import torch

import argparse
from dataset.utils import *
import json
import os
import random
import time
from tqdm import tqdm




def parse_args():
    timestr = time.ctime().replace(f' ', f'-').replace(f':', f'-')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--dataset', default='coco', type=str, 
                        help='data prepare to distillate')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument('--ipc', default=10, type=int, 
                        help='image per class')
    parser.add_argument('--km_expand', default=1, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='/home/xun_ying/DD/data/coco/coco.csv', type=str, 
                        help='root dir')
    parser.add_argument('--prototype_path', default='/home/xun_ying/DD/data/coco/prototypes/coco-ipc10-kmexpand1-Sat-Nov-23-22-41-06-2024.json', type=str, 
                        help='prototype path')
    parser.add_argument('--save_init_image_path', default=os.path.join('/home/xun_ying/DD/data/coco/results/', timestr), type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--strength', '-s', default=0.7, type=float, 
                        help='diffusers strength')
    args = parser.parse_args()
    return args


def load_prototype(args):
    prototype_file_path = args.prototype_path
    with open(prototype_file_path, 'r') as f:
        prototype = json.load(f)
    for prompt, data in prototype.items():
        prototype[prompt] = torch.tensor(data, dtype=torch.float16).to(args.device)
    print("prototype loaded.")
    return prototype


def gen_syn_images(pipe, prototypes, label_list, args):
    for prompt, pros in tqdm(prototypes.items(), total=len(prototypes), position=0):

        assert  args.ipc % pros.size(0) == 0
        for j in range(int(args.ipc/(pros.size(0)))):
            for i in range(pros.size(0)):
                sub_pro = pros[i:i+1]
                sub_pro_random = torch.randn((1, 4, 64, 64), device='cuda',dtype=torch.half)
                negative_prompt = 'cartoon, anime, painting'
                images = pipe(prompt=prompt, latents=sub_pro, negative_prompt=negative_prompt, is_init=True, strength=args.strength, guidance_scale=args.guidance_scale).images
                index = label_list[prompt]
                save_path = os.path.join(args.save_init_image_path, "{}_ipc{}_{}_s{}_g{}_kmexpand{}".format(args.dataset, int(pros.size(0)), args.ipc, args.strength, args.guidance_scale, args.km_expand))
                os.makedirs(os.path.join(save_path, "{}/".format(index)), exist_ok=True)
                images[0].resize((224, 224)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(index, index, i, j)))


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.obtain label-prompt list
    label_dic = gen_label_list(args)

    # 2.define the diffusers pipeline
    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
    )
    pipe = pipe.to(args.device)

    # 3.load prototypes from json file
    prototypes = load_prototype(args)

    # 4.generate initialized synthetic images and save them for refine
    gen_syn_images(pipe=pipe, prototypes=prototypes, label_list={v:k for k,v in label_dic.items()}, args=args)


if __name__ == "__main__" : 
    main()
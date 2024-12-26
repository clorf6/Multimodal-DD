from scripts.pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline

import torch

import argparse
from dataset.utils import *
from torchvision.transforms import ToPILImage
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/nas-new/home/yangnianzu/jsl/Multimodal-DD', type=str, 
                        help='root path')
    parser.add_argument('--dataset', default='coco', type=str, 
                        help='data prepare to distillate')
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5', type=str,
                        help='model name')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument('--time_str', default='Tue-Dec-17-18-00-22-2024', type=str,
                        help='time str')
    parser.add_argument('--config_name', default='coco-res100', type=str,
                        help='configs name')
    parser.add_argument('--strength', '-s', default=0.7, type=float, 
                        help='diffusers strength')
    parser.add_argument("--image_size", default=224, type=int,
                        help="image size")
    parser.add_argument("--image_num", default=5, type=int,
                        help="image number, should be a square number")
    parser.add_argument("--clip_model", default='openai/clip-vit-base-patch32', type=str, 
                        help="clip model name")
    parser.add_argument("--is_resize", default=True, type=bool,
                        help="resize image")
    parser.add_argument("--is_augment", default=False, type=bool,
                        help="use data augmentation")
    parser.add_argument("--is_normalize", default=False, type=bool,
                        help="use data augmentation")
    args = parser.parse_args()
    args.dataset_root = os.path.join(args.root, 'data', args.dataset)
    args.prototype_path = f"{args.root}/results/{args.dataset}/{args.time_str}/prototypes/{args.config_name}.json"
    args.image_path = f"{args.root}/results/{args.dataset}/{args.time_str}/images-new"
    args.save_path = f"{args.root}/results/{args.dataset}/{args.time_str}/results-new.json"
    args.load_path = f"{args.root}/results/{args.dataset}/{args.time_str}/results.json"
    model = args.model.split('/')[-1]
    clip = args.clip_model.split('/')[-1]
    args.model_path = os.path.join(args.root, 'models', model)
    args.clip_path = os.path.join(args.root, 'models', clip)
    os.makedirs(args.image_path, exist_ok=True)
    return args


def load_prototype(args):
    prototype_file_path = args.prototype_path
    with open(prototype_file_path, 'r') as f:
        prototype = tuple(json.load(f))
    prototypes = []
    for (image, caption, idx) in zip(prototype[0], prototype[1], prototype[2]):
        image = torch.tensor(image, dtype=torch.float16).to(args.device)
        caption = torch.tensor(caption, dtype=torch.float16).to(args.device)
        prototypes.append((image, caption, idx))
    print("prototype loaded.")
    return prototypes


def gen_syn_images(pipe, prototypes, dataset, args):
    results = []
    device = args.device
    clip_model, clip_processor = load_clip_model(device, args)
    
    ann = json.load(open(args.load_path, 'r'))
    
    for i, (image_embed, _, idx) in tqdm(enumerate(prototypes), total=len(prototypes)):
        image_embed = image_embed.unsqueeze(0)
        
        _, native_caption, _ = dataset[idx]
        base_caption = ann[i]['caption'][0]
        
        candidate_images = []
        candidate_scores = []
        for _ in range(args.image_num):
            gen_image = pipe(
                latents=image_embed, 
                prompt=base_caption,
                negative_prompt='cartoon, anime, painting',
                is_init=True,
                strength=args.strength,
                guidance_scale=args.guidance_scale
            ).images[0].resize((args.image_size, args.image_size))
            
            score = compute_clip_score(clip_model, clip_processor, gen_image, base_caption, device)
            candidate_images.append(gen_image)
            candidate_scores.append(score)
        
        best_idx = int(np.argmax(candidate_scores))
        best_image = candidate_images[best_idx]
        
        best_image.save(os.path.join(args.image_path, f"{i}.png"))
        img_txt_pair = {'id': i, 'image_path': f"{i}.png", 'caption': [base_caption]}
        results.append(img_txt_pair)
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    # 1.parse args
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.define the diffusers pipeline
    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
    )
    pipe = pipe.to(args.device)

    # 3.load prototypes from json file
    prototypes = load_prototype(args)

    # 4.load training dataset
    train_dataset = load_train_dataset(args)

    # 5.generate initialized synthetic images and save them for refine
    gen_syn_images(pipe=pipe, prototypes=prototypes, dataset=train_dataset, args=args)


if __name__ == "__main__" : 
    main()
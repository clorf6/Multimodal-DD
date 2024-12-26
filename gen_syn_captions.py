import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from dataset.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/nas-new/home/yangnianzu/jsl/Multimodal-DD', type=str, 
                        help='root path')
    parser.add_argument('--model', default='Salesforce/blip2-opt-6.7b-coco', type=str,
                        help='model name')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to distillate')
    parser.add_argument('--time_str', default='Tue-Dec-17-18-00-22-2024', type=str,
                        help='result time str')
    parser.add_argument("--caption_num", default=2, type=int, 
                        help="caption number")
    parser.add_argument("--clip_model", default='openai/clip-vit-base-patch32', type=str, 
                        help="clip model name")
    args = parser.parse_args()
    args.image_path = f"{args.root}/results/{args.dataset}/{args.time_str}/images"
    args.save_path = f"{args.root}/results/{args.dataset}/{args.time_str}/results.json"
    model = args.model.split('/')[-1]
    clip = args.clip_model.split('/')[-1]
    args.model_path = os.path.join(args.root, 'models', model)
    args.clip_path = os.path.join(args.root, 'models', clip)
    return args


def gen_captions(args):
    image_path = args.image_path
    save_path = args.save_path
    
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 加载CLIP
    clip_model, clip_processor = load_clip_model(device, args)

    ann = []
    for file in tqdm(os.listdir(image_path), total=len(os.listdir(image_path))):
        file_path = os.path.join(image_path, file)
        image = Image.open(file_path).convert('RGB')

        num_candidates = args.caption_num
        candidate_captions = []
        candidate_scores = []
        for _ in range(num_candidates):
            inputs = processor(image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            score = compute_clip_score(clip_model, clip_processor, image, generated_text, device)
            candidate_captions.append(generated_text)
            candidate_scores.append(score)
        
        best_idx = int(np.argmax(candidate_scores))
        best_caption = candidate_captions[best_idx]

        img_txt_pair = {'id': int(file.split('.')[0]), 'image_path': file, 'caption': [best_caption]}
        ann.append(img_txt_pair)
            
    ann = sorted(ann, key=lambda x: x['id'])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ann, f, indent=4, ensure_ascii=False)
        
if __name__ == '__main__':
    args = parse_args()
    gen_captions(args)
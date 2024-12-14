import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import argparse
import json
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to distillate')
    parser.add_argument('--time_str', default='Fri-Dec-13-15-24-45-2024', type=str,
                        help='result time str')
    args = parser.parse_args()
    args.image_path = f"/home/xun_ying/DD/results/{args.dataset}/{args.time_str}/images"
    args.save_path = f"/home/xun_ying/DD/results/{args.dataset}/{args.time_str}/results.json"
    return args


def gen_captions(args):
    image_path = args.image_path
    save_path = args.save_path
    
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ann = []
    for file in tqdm(os.listdir(image_path), total=len(os.listdir(image_path))):
        file_path = os.path.join(image_path, file)
        prompt = "" 
        image = Image.open(file_path).convert('RGB')
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = prompt + processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        img_txt_pair = {'id': int(file.split('.')[0]), 'image_path': file_path.replace(image_path, ''), 'caption': [generated_text]}
        ann.append(img_txt_pair)
            
    ann = sorted(ann, key=lambda x: x['id'])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ann, f, indent=4, ensure_ascii=False)
        
if __name__ == '__main__':
    args = parse_args()
    gen_captions(args)
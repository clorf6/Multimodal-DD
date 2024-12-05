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
    parser.add_argument('--prototype_time', default='Thu-Dec--5-08-08-55-2024', type=str,
                        help='prototype time str')
    parser.add_argument('--result_time', default='Thu-Dec--5-20-23-57-2024', type=str,
                        help='result time str')
    parser.add_argument('--config_name', default='coco_ipc1_1_s0.7_g8_kmexpand10', type=str,
                        help='configs name')
    args = parser.parse_args()
    args.label_file_path = f"/home/xun_ying/DD/data/{args.dataset}/prototypes/{args.prototype_time}/label_names.csv"
    args.image_path = f"/home/xun_ying/DD/data/{args.dataset}/results/{args.result_time}/{args.config_name}"
    args.save_path = f"/home/xun_ying/DD/data/{args.dataset}/results/{args.result_time}/syn.json"
    print("label_file_path:", args.label_file_path)
    print("image_path:", args.image_path)
    print("save_path:", args.save_path)
    return args


def gen_captions(args):
    label_file_path = args.label_file_path
    image_path = args.image_path
    save_path = args.save_path
    labels = {}
    df = pd.read_csv(label_file_path)
    labels = dict()
    for _, row in df.iterrows():
        labels[row['id']] = row['name']
    
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    captions = []
    for root, _, files in tqdm(os.walk(image_path)):
        if root == image_path:
            continue
        type = int(root.replace(image_path, '').lstrip('/'))
        prompt = f"This is a picture (maybe ralated to {labels[type]}) of"
        class_captions = {'type': type, 'labels': labels[type], 'image_text_pairs': []}
        for file in files:
            file_path = os.path.join(root, file)
            image = Image.open(file_path).convert('RGB')
            inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            generated_text = prompt + processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            class_captions['image_text_pairs'].append({'image_path': file_path.replace(image_path, ''), 'caption': generated_text})
        captions.append(class_captions)
            
    captions = sorted(captions, key=lambda x: x['type'])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=4, ensure_ascii=False)
        
if __name__ == '__main__':
    args = parse_args()
    gen_captions(args)
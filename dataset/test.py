import json
import numpy as np
from tqdm import tqdm
from collections import Counter

file_path = '/home/xun_ying/DD/data/coco/annotations/train_data.json'  # 替换为你的JSON文件路径
with open(file_path, 'r') as f:
    data = json.load(f)

labels = [item['label'] for item in data["annotations"]]

total_samples = len(labels)
print(f"总样本数量: {total_samples}")

num_classes = 96  
one_hot_encoded = []
for label in tqdm(labels, desc="Processing labels", unit="label"):
    one_hot = np.zeros(num_classes, dtype=int)
    one_hot[label] = 1
    one_hot_encoded.append(tuple(one_hot))  

unique_encoding_counts = Counter(one_hot_encoded)

print(f"不重复的独特编码总数: {len(unique_encoding_counts)}")
sum = 0
tot = 0
for encoding, count in unique_encoding_counts.most_common(500): 
    if count < 10:
        break
    sum += count
    tot += 1
    print(f"编码: {encoding}, 出现次数: {count}")
    
print(f"前1000个最常见的编码总数: {sum}, {tot}, {sum/total_samples*100:.2f}%")
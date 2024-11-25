import json
from collections import defaultdict
from tqdm import tqdm

# 读取JSON文件
with open('/home/xun_ying/DD/data/coco/annotations/captions_val.json', 'r', encoding='utf-8') as f:
    caption_data = json.load(f)

with open('/home/xun_ying/DD/data/coco/annotations/instances_val.json', 'r', encoding='utf-8') as f:
    instance_data = json.load(f)

sorted_data = {}

temp = {}
temp["annotations"] = sorted(caption_data["annotations"], key=lambda x: x["image_id"])

merged_annotations = defaultdict(list)

for annotation in tqdm(temp["annotations"], total=len(temp["annotations"])):
    image_id = annotation.pop("image_id")  # 获取 image_id
    annotation.pop("id", None)  # 删除 id 字段（如果存在）
    merged_annotations[image_id].append(annotation["caption"])  # 按 image_id 分组
    
    
label_list = defaultdict(set)
for sample in tqdm(instance_data["annotations"]):
    label_list[sample["image_id"]].add(sample["category_id"])
    
    
# 创建排序后合并的 annotations 列表
sorted_data["annotations"] = [
    {"image_id": image_id, "captions": annotations, "label": list(label_list[image_id])}
    for image_id, annotations in merged_annotations.items()
]

# 保存到新的JSON文件
with open('/home/xun_ying/DD/data/coco/annotations/val_data.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_data, f, indent=4, ensure_ascii=False)

